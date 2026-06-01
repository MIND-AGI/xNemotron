from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cosmos_xenna.pipelines.v1 as pipelines_v1
import numpy as np

from nemotron.data_prep.core.chat_sft_shard_core import (
    process_chat_sft_spool_core,
    process_chat_sft_unpacked_parquet_from_spool_core,
)
from nemotron.data_prep.core.receipt import ReceiptManager
from nemotron.data_prep.core.work_items import SftShardWorkItem
from nemotron.data_prep.stages.context import PipelineContext
from nemotron.data_prep.utils.filesystem import get_filesystem


@dataclass(frozen=True)
class UnpackedSftParquetStageConfig:
    """Configuration for un-packed SFT Parquet export."""

    cpus_per_worker: int = 4

    def __post_init__(self) -> None:
        if self.cpus_per_worker <= 0:
            raise ValueError(f"cpus_per_worker must be positive, got {self.cpus_per_worker}")


class UnpackedSftParquetStage(pipelines_v1.Stage[SftShardWorkItem, SftShardWorkItem]):
    """Shard processing stage that exports one row per original sequence."""

    def __init__(
        self,
        stage_config: UnpackedSftParquetStageConfig,
        pipeline_context: PipelineContext,
    ) -> None:
        if pipeline_context.resolved_tokenizer is None:
            raise ValueError("UnpackedSftParquetStage requires resolved_tokenizer in PipelineContext")
        if pipeline_context.run_hash is None:
            raise ValueError("UnpackedSftParquetStage requires run_hash in PipelineContext")

        self._cfg = stage_config
        self._ctx = pipeline_context
        self._tokenizer = None
        self._fs = None
        self._receipts: ReceiptManager | None = None

    @property
    def stage_batch_size(self) -> int:
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=self._cfg.cpus_per_worker, gpus=0)

    @property
    def env_info(self) -> pipelines_v1.RuntimeEnv:
        return self._ctx.hf_runtime_env()

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._ctx.resolved_tokenizer["model"],
            revision=self._ctx.resolved_tokenizer.get("resolved_revision"),
            trust_remote_code=self._ctx.resolved_tokenizer.get("trust_remote_code", False),
            local_files_only=True,
        )
        self._fs, _ = get_filesystem(self._ctx.output_root)
        self._receipts = ReceiptManager(self._fs, self._ctx.run_hash)

    def process_data(self, tasks: list[SftShardWorkItem]) -> list[None]:
        for task in tasks:
            self._process_shard(task)
        return []

    def _process_shard(self, task: SftShardWorkItem) -> None:
        receipts = self._get_receipts()
        rpath = receipts.receipt_path(task.receipts_dir, task.shard_index)

        if receipts.is_completed(
            rpath,
            task.plan_hash,
            verify_outputs=lambda: self._outputs_exist(task),
        ):
            return

        meta = dict(
            plan_hash=task.plan_hash,
            shard_index=task.shard_index,
            dataset_name=task.dataset_name,
        )
        receipts.write_started(rpath, **meta)

        try:
            self._run_spool(task)
            stats, files = self._build_completed_payload(task)
            receipts.write_completed(rpath, stats=stats, files=files, **meta)
        except Exception as e:
            receipts.write_failed(rpath, error=e, **meta)
            raise

    def _outputs_exist(self, task: SftShardWorkItem) -> bool:
        from nemotron.data_prep.utils.filesystem import read_json

        rpath = self._get_receipts().receipt_path(task.receipts_dir, task.shard_index)
        try:
            receipt = read_json(self._fs, rpath)
            stats = receipt.get("stats", {}) or {}
            if int(stats.get("num_sequences", 0) or 0) == 0:
                return True
            parquet_rel = ((receipt.get("files", {}).get("parquet") or {}).get("path")) or ""
            if not parquet_rel:
                return False
            return self._fs.exists(f"{task.output_dir.rstrip('/')}/{parquet_rel}")
        except Exception:
            return False

    def _build_completed_payload(self, task: SftShardWorkItem) -> tuple[dict[str, Any], dict[str, Any]]:
        return process_chat_sft_unpacked_parquet_from_spool_core(
            shard_index=task.shard_index,
            output_dir=task.output_dir,
            spool_dir=self._resolve_spool_dir(task),
            output_fs=self._fs,
            parquet_row_group_size=int(task.parquet_row_group_size),
            parquet_compression=str(task.parquet_compression),
        )

    def _resolve_spool_dir(self, task: SftShardWorkItem) -> str:
        if task.spool_dir:
            return task.spool_dir
        shard_id = f"shard_{task.shard_index:06d}"
        return f"{task.output_dir.rstrip('/')}/spool/{shard_id}"

    def _run_spool(self, task: SftShardWorkItem) -> None:
        spool_dir = self._resolve_spool_dir(task)

        process_chat_sft_spool_core(
            shard_index=task.shard_index,
            files=task.assignment.get("files", []),
            output_dir=task.output_dir,
            receipts_dir=task.receipts_dir,
            spool_dir=spool_dir,
            output_fs=self._fs,
            tokenizer=self._tokenizer,
            messages_field=task.messages_field,
            tools_field=task.tools_field,
            pack_size=int(task.pack_size),
            algorithm=str(task.algorithm),
            dtype=np.dtype(task.dtype),
            chat_template=task.chat_template,
            max_doc_tokens=task.max_doc_tokens,
            max_rows=task.max_rows,
            seed=task.seed,
            used_in_filter=task.used_in_filter,
            used_in_field=task.used_in_field,
        )

    def _get_receipts(self) -> ReceiptManager:
        if self._receipts is None:
            self._receipts = ReceiptManager(self._fs, self._ctx.run_hash)
        return self._receipts


__all__ = ["UnpackedSftParquetStage", "UnpackedSftParquetStageConfig"]
