#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "nano3/data/export/unpacked-parquet"
# image = "anyscale/ray:2.49.2-py312"
# setup = """
# Requires the full nemotron repository synced to the worker.
# Install the nemotron package with xenna extras: uv sync --reinstall-package nemotron.
# """
#
# [tool.runspec.run]
# launch = "ray"
# cmd = "uv run --extra xenna python {script} --config {config}"
#
# [tool.runspec.config]
# dir = "./config/data_prep"
# default = "unpacked_parquet"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
# ///

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cosmos_xenna.pipelines.v1 as pipelines_v1

from nemotron.data_prep.blend import DataBlend
from nemotron.data_prep.config import ObservabilityConfig, TokenizerConfig
from nemotron.data_prep.core.finalize import scan_dataset_receipts
from nemotron.data_prep.observability import pipeline_wandb_hook
from nemotron.data_prep.recipes.execution_mode import resolve_execution_mode
from nemotron.data_prep.recipes.sft import SftPlanAdapter, setup_sft_run
from nemotron.data_prep.stages import (
    DownloadStage,
    DownloadStageConfig,
    PipelineContext,
    PlanStage,
    SftPlanStageConfig,
    UnpackedSftParquetStage,
    UnpackedSftParquetStageConfig,
)
from nemotron.data_prep.utils.filesystem import get_filesystem, write_json
from nemotron.data_prep.utils.hf_env import detect_hf_env_vars
from nemotron.kit import wandb_kit
from nemotron.kit.train_script import (
    apply_hydra_overrides,
    init_wandb_from_env,
    load_omegaconf_yaml,
    omegaconf_to_dataclass,
    parse_config_and_overrides,
)

logger = logging.getLogger(__name__)

STAGE_PATH = Path(__file__).parent
DEFAULT_CONFIG_PATH = STAGE_PATH / "config" / "data_prep" / "unpacked_parquet.yaml"
_OUTPUT_BASE = Path(os.environ.get("NEMO_RUN_DIR", "."))
RAY = True


@dataclass
class UnpackedParquetExportConfig:
    """Export chat-tokenized, un-packed parquet with labels/input_ids/loss_mask."""

    blend_path: Path = field(default_factory=lambda: STAGE_PATH / "config/data_prep/data_blend_raw.json")
    output_dir: Path = field(default_factory=lambda: _OUTPUT_BASE / "stage1_sft_unpacked")
    num_shards: int = 128

    tokenizer: TokenizerConfig = field(default_factory=lambda: TokenizerConfig(
        model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
        add_bos=False,
        add_eos=True,
    ))

    parquet_row_group_size: int = 1000
    parquet_compression: str = "zstd"

    chat_template: str = "nano3"
    messages_field: str = "messages"
    tools_field: str = "tools"
    used_in_filter: str | None = None
    used_in_field: str = "used_in"
    max_doc_tokens: int | None = None

    sample: int | None = None
    sample_seed: int = 42
    force: bool = False
    execution_mode: str = "auto"
    config_name: str = "default"

    # Reuse shared SFT setup/work-item machinery. These fields are not used for
    # parquet packing, but remain in the hashed run config and spool manifest.
    pack_size: int = 4096
    algorithm: str = "first_fit_shuffle"
    seed: int | None = None

    plan: SftPlanStageConfig = field(default_factory=SftPlanStageConfig)
    download: DownloadStageConfig = field(default_factory=DownloadStageConfig)
    tokenization: UnpackedSftParquetStageConfig = field(default_factory=UnpackedSftParquetStageConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    def __post_init__(self) -> None:
        if isinstance(self.blend_path, str):
            self.blend_path = Path(self.blend_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if self.sample is not None:
            self.output_dir = self.output_dir / f"sample-{self.sample}"


def _build_export_manifest(
    *,
    run_dir: str,
    output_dir: Path,
    dataset_names: list[str],
) -> Path:
    fs, _ = get_filesystem(str(output_dir))
    scanned = scan_dataset_receipts(run_dir, dataset_names, fs)
    datasets: dict[str, dict[str, object]] = {}
    total_rows = 0
    total_tokens = 0

    for dataset_name, receipts in scanned.items():
        parquet_files = []
        dataset_rows = 0
        dataset_tokens = 0
        for receipt in receipts.completed:
            parquet_rel = ((receipt.get("files", {}).get("parquet") or {}).get("path")) or ""
            if parquet_rel:
                parquet_files.append(f"{run_dir}/datasets/{dataset_name}/{receipts.plan_hash}/{parquet_rel}")
            stats = receipt.get("stats", {}) or {}
            dataset_rows += int(stats.get("num_rows", stats.get("num_sequences", 0)) or 0)
            dataset_tokens += int(stats.get("total_tokens", 0) or 0)

        datasets[dataset_name] = {
            "plan_hash": receipts.plan_hash,
            "num_shards_completed": len(receipts.completed),
            "num_rows": dataset_rows,
            "total_tokens": dataset_tokens,
            "parquet_files": parquet_files,
        }
        total_rows += dataset_rows
        total_tokens += dataset_tokens

    manifest = {
        "format": "unpacked_sft_parquet",
        "run_dir": run_dir,
        "total_rows": total_rows,
        "total_tokens": total_tokens,
        "datasets": datasets,
    }
    manifest_path = output_dir / "manifest.json"
    write_json(fs, str(manifest_path), manifest)
    return manifest_path


def run_export_main(cfg: UnpackedParquetExportConfig) -> Path:
    start_time = time.time()
    wandb_kit.add_run_tags(["data-prep", "sft", "unpacked-parquet"])
    wandb_kit.log_wandb_config(cfg)

    blend = DataBlend.load(cfg.blend_path)
    if blend.datasets is None:
        raise ValueError(
            f"Expected single-blend format (datasets list), but got per-split blend from {cfg.blend_path}"
        )

    num_shards_effective = 1 if cfg.sample is not None else cfg.num_shards
    packing_seed = cfg.seed if cfg.seed is not None else cfg.sample_seed

    dataset_items, context, resolved_tokenizer = setup_sft_run(
        blend=blend,
        output_dir=cfg.output_dir,
        tokenizer=cfg.tokenizer,
        num_shards=num_shards_effective,
        messages_field_default=cfg.messages_field,
        tools_field_default=cfg.tools_field,
        chat_template=cfg.chat_template,
        used_in_filter=cfg.used_in_filter,
        used_in_field=cfg.used_in_field,
        pack_size=cfg.pack_size,
        algorithm=cfg.algorithm,
        seed=packing_seed,
        parquet_row_group_size=cfg.parquet_row_group_size,
        parquet_compression=cfg.parquet_compression,
        max_doc_tokens=cfg.max_doc_tokens,
        max_rows=cfg.sample,
        sample_seed=cfg.sample_seed,
        force=cfg.force,
    )

    if dataset_items:
        pipeline_ctx = PipelineContext(
            output_root=str(cfg.output_dir),
            run_hash=context.run_hash,
            run_dir=context.run_dir,
            config_hash=None,
            resolved_tokenizer=resolved_tokenizer,
            observability=cfg.observability,
            hf_env=detect_hf_env_vars(),
        )
        stage_specs = [
            pipelines_v1.StageSpec(PlanStage(cfg.plan, pipeline_ctx, SftPlanAdapter()), num_workers=1),
            pipelines_v1.StageSpec(DownloadStage(cfg.download, pipeline_ctx), num_workers_per_node=1),
            pipelines_v1.StageSpec(UnpackedSftParquetStage(cfg.tokenization, pipeline_ctx), slots_per_actor=1),
        ]
        spec = pipelines_v1.PipelineSpec(
            input_data=dataset_items,
            stages=stage_specs,
            config=pipelines_v1.PipelineConfig(
                execution_mode=resolve_execution_mode(stage_specs, cfg.execution_mode),
            ),
        )
        with pipeline_wandb_hook(dataset_items, pipeline_ctx, "sft-unpacked-parquet"):
            pipelines_v1.run_pipeline(spec)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _build_export_manifest(
        run_dir=context.run_dir,
        output_dir=cfg.output_dir,
        dataset_names=context.dataset_names,
    )

    elapsed = time.time() - start_time
    logger.info("Finished exporting un-packed parquet in %.2fs: %s", elapsed, manifest_path)
    wandb_kit.finish_run(exit_code=0)
    return manifest_path


def main(cfg: UnpackedParquetExportConfig | None = None) -> Path:
    if cfg is None:
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)
        try:
            config = load_omegaconf_yaml(config_path)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if cli_overrides:
            config = apply_hydra_overrides(config, cli_overrides)

        cfg = omegaconf_to_dataclass(config, UnpackedParquetExportConfig)

    init_wandb_from_env()
    manifest_path = run_export_main(cfg)
    print(json.dumps({"manifest": str(manifest_path)}, indent=2))
    return manifest_path


if __name__ == "__main__":
    main()
