from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
from fsspec.implementations.local import LocalFileSystem

from nemotron.data_prep.core.chat_sft_shard_core import (
    _build_unpacked_labels_and_loss_mask,
    process_chat_sft_unpacked_parquet_from_spool_core,
)
from nemotron.data_prep.packing.spool import SequenceSpoolPaths, SequenceSpoolWriter


def test_build_unpacked_labels_and_loss_mask() -> None:
    input_ids = np.asarray([10, 11, 12, 13], dtype=np.int32)
    original_loss_mask = np.asarray([0, 0, 1, 1], dtype=np.uint8)

    labels, aligned_loss_mask = _build_unpacked_labels_and_loss_mask(input_ids, original_loss_mask)

    assert labels.tolist() == [-100, 12, 13, -100]
    assert aligned_loss_mask.tolist() == [0, 1, 1, 0]


def test_process_chat_sft_unpacked_parquet_from_spool_core(tmp_path) -> None:
    fs = LocalFileSystem()
    spool_root = tmp_path / "spool" / "shard_000000"
    spool_paths = SequenceSpoolPaths.for_root(str(spool_root))
    writer = SequenceSpoolWriter(fs=fs, paths=spool_paths)

    writer.append(
        np.asarray([101, 102, 103, 104], dtype=np.int32),
        np.asarray([0, 0, 1, 1], dtype=np.uint8),
    )
    writer.append(
        np.asarray([201], dtype=np.int32),
        np.asarray([1], dtype=np.uint8),
    )
    writer.finalize(
        extra_manifest={
            "tokenization_stats": {
                "num_input_rows": 2,
                "num_output_sequences": 2,
            },
            "input_files": ["dummy.jsonl"],
        }
    )

    output_dir = tmp_path / "out"
    stats, files = process_chat_sft_unpacked_parquet_from_spool_core(
        shard_index=0,
        output_dir=str(output_dir),
        spool_dir=str(spool_root),
        output_fs=fs,
        parquet_row_group_size=16,
        parquet_compression="none",
    )

    parquet_path = output_dir / "shard_000000.parquet"
    rows = pq.read_table(parquet_path).to_pylist()

    assert parquet_path.exists()
    assert files["parquet"]["path"] == "shard_000000.parquet"
    assert stats["num_sequences"] == 2
    assert stats["num_rows"] == 2
    assert stats["total_tokens"] == 5
    assert stats["num_supervised_tokens"] == 2

    assert rows[0] == {
        "input_ids": [101, 102, 103, 104],
        "loss_mask": [0, 1, 1, 0],
        "labels": [-100, 103, 104, -100],
    }
    assert rows[1] == {
        "input_ids": [201],
        "loss_mask": [0],
        "labels": [-100],
    }
