#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pyarrow.parquet as pq
from transformers import AutoTokenizer


DEFAULT_TOKENIZER = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
ROLE_MARKER_RE = re.compile(r"<\|im_start\|>([A-Za-z_][\w-]*)\n")


@dataclass
class FormState:
    parquet_path: str = ""
    tokenizer: str = DEFAULT_TOKENIZER
    hf_home: str = ""
    local_files_only: bool = True
    trust_remote_code: bool = True


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def parse_bool(values: dict[str, list[str]], key: str) -> bool:
    return values.get(key, [""])[0] in {"1", "true", "on", "yes"}


def form_from_values(values: dict[str, list[str]] | None = None) -> FormState:
    if values is None:
        return FormState()
    return FormState(
        parquet_path=values.get("parquet_path", [""])[0],
        tokenizer=values.get("tokenizer", [DEFAULT_TOKENIZER])[0],
        hf_home=values.get("hf_home", [""])[0],
        local_files_only=parse_bool(values, "local_files_only"),
        trust_remote_code=parse_bool(values, "trust_remote_code"),
    )


def resolve_tokenizer_source(source: str) -> str:
    """Resolve repo id, local tokenizer dir, or HF hub cache model dir.

    AutoTokenizer accepts a repo id such as ``nvidia/foo`` or a local snapshot
    directory containing config/tokenizer files. It does not reliably accept the
    cache root ``.../hub/models--org--repo`` directly, so resolve that form.
    """
    if not source:
        raise ValueError("tokenizer must not be empty")

    path = Path(source).expanduser()
    if not path.exists():
        return source

    if path.is_file():
        raise ValueError(f"tokenizer must be a repo id or directory, got file: {path}")

    direct_markers = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "tokenizer.model",
    ]
    if any((path / marker).exists() for marker in direct_markers):
        return str(path)

    refs_main = path / "refs" / "main"
    snapshots_dir = path / "snapshots"
    if refs_main.exists() and snapshots_dir.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot = snapshots_dir / revision
        if snapshot.exists():
            return str(snapshot)

    if snapshots_dir.exists():
        snapshots = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
        if len(snapshots) == 1:
            return str(snapshots[0])
        if snapshots:
            raise ValueError(
                "tokenizer cache root has multiple snapshots; pass one snapshot path explicitly: "
                + ", ".join(str(p) for p in snapshots[:5])
            )

    raise ValueError(
        "tokenizer directory is not a loadable snapshot. Pass a repo id or a directory containing "
        "config.json/tokenizer_config.json/tokenizer.json, e.g. .../snapshots/<revision>."
    )


def decode_one(tokenizer, token_id: int | None) -> str:
    if token_id is None:
        return ""
    return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)


def sequence_bounds(seq_start_id: list[int], n_tokens: int) -> list[tuple[int, int]]:
    starts = [int(x) for x in seq_start_id if 0 <= int(x) < n_tokens]
    if not starts:
        return [(0, n_tokens)]
    if starts[0] != 0:
        starts = [0] + starts
    starts = sorted(set(starts))
    return list(zip(starts, starts[1:] + [n_tokens]))


def load_parquet_sequences(path: str) -> dict:
    table = pq.read_table(path)
    data = table.to_pydict()
    has_labels = "labels" in data
    has_seq_start_id = "seq_start_id" in data
    sequences = []
    for row_index in range(table.num_rows):
        input_ids = [int(x) for x in data["input_ids"][row_index]]
        loss_mask = [int(x) for x in data["loss_mask"][row_index]]
        labels = [int(x) for x in data["labels"][row_index]] if has_labels else []
        # Un-packed parquet stores one logical sequence per row and therefore
        # does not carry seq_start_id. Treat the whole row as a single sequence.
        seq_start_id = [int(x) for x in data["seq_start_id"][row_index]] if has_seq_start_id else [0]
        bounds = sequence_bounds(seq_start_id, len(input_ids))
        for seq_index, (start, end) in enumerate(bounds):
            sequences.append(
                {
                    "row_index": row_index,
                    "seq_index": seq_index,
                    "seq_start": start,
                    "seq_end": end,
                    "input_ids": input_ids[start:end],
                    "loss_mask": loss_mask[start:end],
                    "labels": labels[start:end] if has_labels else [],
                    "packed_len": len(input_ids),
                    "seq_start_id": seq_start_id,
                    "bounds": bounds,
                }
            )
    return {
        "table_rows": table.num_rows,
        "schema": str(table.schema),
        "has_seq_start_id": has_seq_start_id,
        "sequences": sequences,
    }


def build_parquet_role_segments(tokenizer, actual: dict, decoded_text: str) -> list[dict]:
    matches = list(ROLE_MARKER_RE.finditer(decoded_text))
    if not matches:
        return [
            {
                "role": "unknown",
                "text": decoded_text,
                "token_start": 0,
                "token_end": len(actual["input_ids"]),
                "input_ids": actual["input_ids"],
                "loss_mask": actual["loss_mask"],
                "labels": actual.get("labels", []),
                "seq_start_id": actual["seq_start_id"],
            }
        ]

    segments = []
    token_pos = 0
    for idx, match in enumerate(matches):
        text_start = match.start()
        text_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(decoded_text)
        text = decoded_text[text_start:text_end]
        encoded = tokenizer.encode(text, add_special_tokens=False)
        start = token_pos
        end = min(start + len(encoded), len(actual["input_ids"]))
        segments.append(
            {
                "role": match.group(1),
                "text": text,
                "token_start": start,
                "token_end": end,
                "input_ids": actual["input_ids"][start:end],
                "loss_mask": actual["loss_mask"][start:end],
                "labels": actual.get("labels", [])[start:end],
                "seq_start_id": actual["seq_start_id"],
            }
        )
        token_pos = end

    if token_pos < len(actual["input_ids"]):
        segments.append(
            {
                "role": "tail",
                "text": tokenizer.decode(actual["input_ids"][token_pos:], clean_up_tokenization_spaces=False),
                "token_start": token_pos,
                "token_end": len(actual["input_ids"]),
                "input_ids": actual["input_ids"][token_pos:],
                "loss_mask": actual["loss_mask"][token_pos:],
                "labels": actual.get("labels", [])[token_pos:],
                "seq_start_id": actual["seq_start_id"],
            }
        )
    return segments


def role_at(segments: list[dict], pos: int) -> str:
    for segment in segments:
        if segment["token_start"] <= pos < segment["token_end"]:
            return str(segment["role"])
    return ""


def build_parquet_token_rows(tokenizer, actual: dict, role_segments: list[dict]) -> list[dict]:
    pq_ids = actual["input_ids"]
    pq_mask = actual["loss_mask"]
    pq_labels = actual.get("labels", [])
    rows = []
    for pos, pq_id in enumerate(pq_ids):
        actual_mask = pq_mask[pos] if pos < len(pq_mask) else None
        inferred_label_id = pq_ids[pos + 1] if pos + 1 < len(pq_ids) else None
        stored_label_id = pq_labels[pos] if pos < len(pq_labels) else None
        rows.append(
            {
                "pos": pos,
                "role": role_at(role_segments, pos),
                "pq_id": pq_id,
                "pq_text": decode_one(tokenizer, pq_id),
                "pq_loss_mask": actual_mask,
                "stored_label_id": stored_label_id,
                "stored_label_text": decode_one(tokenizer, stored_label_id)
                if stored_label_id not in {None, -100}
                else "",
                "inferred_label_id": inferred_label_id,
                "inferred_label_text": decode_one(tokenizer, inferred_label_id),
                "trained_label": (
                    decode_one(tokenizer, stored_label_id)
                    if stored_label_id not in {None, -100}
                    else ""
                ),
                "label_match": stored_label_id in {None, -100} or stored_label_id == inferred_label_id,
            }
        )
    return rows


def summarize_sequence(actual: dict, role_segments: list[dict]) -> dict:
    labels = actual.get("labels", [])
    return {
        "parquet_seq_tokens": len(actual["input_ids"]),
        "trainable_labels": sum(actual["loss_mask"]),
        "stored_labels": sum(1 for label in labels if label != -100),
        "role_segments": len(role_segments),
        "has_role_markers": any(segment["role"] != "unknown" for segment in role_segments),
    }


def analyze(state: FormState) -> dict:
    if state.hf_home:
        os.environ["HF_HOME"] = state.hf_home

    tokenizer_source = resolve_tokenizer_source(state.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        local_files_only=state.local_files_only,
        trust_remote_code=state.trust_remote_code,
    )
    parquet = load_parquet_sequences(state.parquet_path)
    parquet_sequences = parquet["sequences"]

    comparisons = []
    for sequence_index, actual in enumerate(parquet_sequences):
        decoded_parquet = tokenizer.decode(actual["input_ids"], clean_up_tokenization_spaces=False)
        role_segments = build_parquet_role_segments(tokenizer, actual, decoded_parquet)
        summary = summarize_sequence(actual, role_segments)
        rows = build_parquet_token_rows(tokenizer, actual, role_segments)
        comparisons.append(
            {
                "sequence_index": sequence_index,
                "actual": actual,
                "summary": summary,
                "rows": rows,
                "role_segments": role_segments,
                "decoded_parquet": decoded_parquet,
            }
        )

    return {
        "parquet": parquet,
        "comparisons": comparisons,
        "summary": {
            "parquet_rows": parquet["table_rows"],
            "parquet_sequences": len(parquet_sequences),
            "role_segments": sum(item["summary"]["role_segments"] for item in comparisons),
            "sequences_with_role_markers": sum(
                1 for item in comparisons if item["summary"]["has_role_markers"]
            ),
            "trainable_labels": sum(item["summary"]["trainable_labels"] for item in comparisons),
            "stored_labels": sum(item["summary"]["stored_labels"] for item in comparisons),
        },
        "tokenizer_source": tokenizer_source,
    }


STYLE = """
<style>
:root {
  color-scheme: dark;
  --bg: #0b1117;
  --panel: #111b24;
  --panel-2: #162331;
  --ink: #e8f1f7;
  --muted: #8ca3b4;
  --line: #26394a;
  --good: #57d68d;
  --bad: #ff6b6b;
  --warn: #ffd166;
  --accent: #7dd3fc;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background:
    radial-gradient(circle at 18% 8%, rgba(125, 211, 252, .16), transparent 28%),
    radial-gradient(circle at 80% 0%, rgba(87, 214, 141, .10), transparent 26%),
    var(--bg);
  color: var(--ink);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
main { width: min(1480px, calc(100vw - 48px)); margin: 32px auto 80px; }
h1 { font-size: 28px; letter-spacing: -0.04em; margin: 0 0 8px; }
h2 { font-size: 18px; margin: 0 0 16px; }
p { color: var(--muted); line-height: 1.6; }
.grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
.span2 { grid-column: span 2; }
.span4 { grid-column: span 4; }
.card {
  border: 1px solid var(--line);
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(255,255,255,.035), rgba(255,255,255,.015)), var(--panel);
  box-shadow: 0 18px 70px rgba(0,0,0,.32);
  padding: 18px;
  margin-top: 18px;
}
label { display: block; color: var(--muted); font-size: 12px; margin-bottom: 6px; }
input[type="text"], input[type="number"] {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #0b141d;
  color: var(--ink);
  padding: 11px 12px;
  outline: none;
}
input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(125,211,252,.12); }
.check { display: flex; align-items: center; gap: 8px; margin-top: 22px; color: var(--muted); }
button {
  border: 0;
  border-radius: 12px;
  background: linear-gradient(135deg, #7dd3fc, #57d68d);
  color: #061018;
  padding: 12px 18px;
  font-weight: 800;
  cursor: pointer;
}
.summary { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; }
.metric { background: var(--panel-2); border: 1px solid var(--line); border-radius: 14px; padding: 12px; }
.metric b { display: block; font-size: 20px; }
.ok { color: var(--good); }
.bad { color: var(--bad); }
.warn { color: var(--warn); }
pre {
  overflow: auto;
  white-space: pre-wrap;
  background: #071019;
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 14px;
  color: #d7e7f2;
  max-height: 420px;
}
pre.role-pre {
  max-height: none;
  min-width: 260px;
}
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th {
  position: sticky;
  top: 0;
  background: #10202d;
  color: var(--accent);
  text-align: left;
  z-index: 1;
}
th, td { border-bottom: 1px solid var(--line); padding: 8px; vertical-align: top; }
tr.mismatch { background: rgba(255, 107, 107, .08); }
tr.maskonly { background: rgba(255, 209, 102, .07); }
.scroll { max-height: 720px; overflow: auto; border: 1px solid var(--line); border-radius: 14px; }
.pill { display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 3px 8px; color: var(--muted); }
details.seq, details.role-group {
  border: 1px solid var(--line);
  border-radius: 14px;
  background: rgba(255,255,255,.02);
  margin-top: 14px;
}
details.seq > summary, details.role-group > summary {
  cursor: pointer;
  padding: 12px 14px;
  color: var(--ink);
  font-weight: 700;
}
details.role-group { margin: 10px 0; }
details.role-group > summary { color: var(--accent); }
.seq-body { padding: 0 14px 14px; }
</style>
"""


def render_form(state: FormState) -> str:
    checked_local = "checked" if state.local_files_only else ""
    checked_trust = "checked" if state.trust_remote_code else ""
    return f"""
<form method="post" class="card">
  <h2>输入</h2>
  <div class="grid">
    <div class="span2">
      <label>Parquet 文件路径</label>
      <input name="parquet_path" type="text" value="{esc(state.parquet_path)}" placeholder="/path/to/shard_000000.parquet">
    </div>
    <div class="span2">
      <label>Tokenizer 名称或本地路径</label>
      <input name="tokenizer" type="text" value="{esc(state.tokenizer)}">
    </div>
    <div>
      <label>HF_HOME 可选</label>
      <input name="hf_home" type="text" value="{esc(state.hf_home)}">
    </div>
    <div class="check">
      <input id="local_files_only" name="local_files_only" type="checkbox" {checked_local}>
      <label for="local_files_only">local_files_only</label>
    </div>
    <div class="check">
      <input id="trust_remote_code" name="trust_remote_code" type="checkbox" {checked_trust}>
      <label for="trust_remote_code">trust_remote_code</label>
    </div>
    <div class="span4">
      <button type="submit">开始查看 Parquet</button>
    </div>
  </div>
</form>
"""


def render_rows_table(rows: list[dict]) -> str:
    body = []
    for row in rows:
        pq_text = "" if row["pq_id"] is None else repr(row["pq_text"])
        stored_label_text = "" if not row["stored_label_text"] else repr(row["stored_label_text"])
        inferred_label_text = "" if row["inferred_label_id"] is None else repr(row["inferred_label_text"])
        trained_label = "" if not row["trained_label"] else repr(row["trained_label"])
        stored_label_id = "" if row["stored_label_id"] is None else row["stored_label_id"]
        inferred_label_id = "" if row["inferred_label_id"] is None else row["inferred_label_id"]
        label_match_class = "ok" if row["label_match"] else "bad"
        label_match_text = "OK" if row["label_match"] else "DIFF"
        body.append(
            f"""
<tr>
  <td>{row["pos"]}</td>
  <td><span class="pill">{esc(row["role"])}</span></td>
  <td>{esc(row["pq_id"])}</td>
  <td>{esc(pq_text)}</td>
  <td>{esc(row["pq_loss_mask"])}</td>
  <td>{esc(stored_label_id)}</td>
  <td>{esc(stored_label_text)}</td>
  <td>{esc(inferred_label_id)}</td>
  <td>{esc(inferred_label_text)}</td>
  <td>{esc(trained_label)}</td>
  <td class="{label_match_class}">{label_match_text}</td>
</tr>
"""
        )
    return f"""
<div class="scroll">
  <table>
    <thead>
      <tr>
        <th>pos</th><th>role</th><th>pq_id</th><th>pq_text</th><th>pq_mask</th>
        <th>labels[i]</th><th>labels_text</th><th>next_id</th><th>next_text</th><th>trained_label</th><th>label</th>
      </tr>
    </thead>
    <tbody>{''.join(body)}</tbody>
  </table>
</div>
"""


def group_rows_by_role(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    groups: list[tuple[str, list[dict]]] = []
    for row in rows:
        role = row.get("role") or "unknown"
        if not groups or groups[-1][0] != role:
            groups.append((role, [row]))
        else:
            groups[-1][1].append(row)
    return groups


def render_collapsible_rows(rows: list[dict]) -> str:
    groups_html = []
    for role, group in group_rows_by_role(rows):
        start = group[0]["pos"]
        end = group[-1]["pos"]
        trainable_labels = sum(1 for row in group if row["pq_loss_mask"] == 1)
        stored_labels = sum(1 for row in group if row["stored_label_id"] not in {None, -100})
        groups_html.append(
            f"""
<details class="role-group">
  <summary>{esc(role)} | pos {start}-{end} | tokens {len(group)} | trainable_labels {trainable_labels} | stored_labels {stored_labels}</summary>
  {render_rows_table(group)}
</details>
"""
        )
    return "".join(groups_html)


def render_json_value(value: object) -> str:
    return esc(json.dumps(value, ensure_ascii=False))


def render_role_segments_table(segments: list[dict]) -> str:
    body = []
    for idx, segment in enumerate(segments):
        body.append(
            f"""
<tr>
  <td>{idx}</td>
  <td><span class="pill">{esc(segment.get("role", ""))}</span></td>
  <td>{esc(segment.get("token_start", ""))}-{esc(segment.get("token_end", ""))}</td>
  <td><pre class="role-pre">{esc(segment.get("text", ""))}</pre></td>
  <td><pre class="role-pre">{render_json_value(segment.get("input_ids", []))}</pre></td>
  <td><pre class="role-pre">{render_json_value(segment.get("loss_mask", []))}</pre></td>
  <td><pre class="role-pre">{render_json_value(segment.get("labels", []))}</pre></td>
  <td><pre class="role-pre">{render_json_value(segment.get("seq_start_id", []))}</pre></td>
</tr>
"""
        )
    return f"""
<div class="scroll">
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>role</th>
        <th>token_range</th>
        <th>text</th>
        <th>input_ids</th>
        <th>loss_mask</th>
        <th>labels</th>
        <th>seq_start_id</th>
      </tr>
    </thead>
    <tbody>{''.join(body)}</tbody>
  </table>
</div>
"""


def render_report(result: dict) -> str:
    summary = result["summary"]
    train_ok = "ok" if summary["trainable_labels"] > 0 else "bad"
    sequence_sections = []
    for item in result["comparisons"]:
        seq_summary = item["summary"]
        actual = item["actual"]
        role_class = "ok" if seq_summary["has_role_markers"] else "warn"
        open_attr = " open" if item["sequence_index"] == 0 else ""
        sequence_sections.append(
            f"""
<details class="seq"{open_attr}>
  <summary>
    sequence {item["sequence_index"]} |
    parquet row {actual["row_index"]} seq {actual["seq_index"]} |
    tokens {seq_summary["parquet_seq_tokens"]} |
    role_segments {seq_summary["role_segments"]} |
    role_markers <span class="{role_class}">{seq_summary["has_role_markers"]}</span> |
    trainable_labels {seq_summary["trainable_labels"]} |
    stored_labels {seq_summary["stored_labels"]}
  </summary>
  <div class="seq-body">
    <h2>Parquet 反解码</h2>
    <pre>{esc(item["decoded_parquet"])}</pre>
    <h2>Role 段落表</h2>
    <p>此表按 `&lt;|im_start|&gt;role` 标记切分；若未识别到标记，会整段显示为 `unknown`。每段同时展示对应的 `labels` 切片。</p>
    {render_role_segments_table(item["role_segments"])}
    <h2>逐 token 表</h2>
    <p>此表按连续 role 折叠，展示 token、loss_mask、parquet 原生 `labels[i]`，以及按 `input_ids[i + 1]` 推导出的 next-token label。</p>
    {render_collapsible_rows(item["rows"])}
  </div>
</details>
"""
        )

    parquet = result["parquet"]
    return f"""
<section class="card">
  <h2>概览</h2>
  <div class="summary">
    <div class="metric"><span>Parquet rows</span><b>{summary["parquet_rows"]}</b></div>
    <div class="metric"><span>Parquet sequences</span><b>{summary["parquet_sequences"]}</b></div>
    <div class="metric"><span>Role segments</span><b>{summary["role_segments"]}</b></div>
    <div class="metric"><span>带 role marker 的序列</span><b>{summary["sequences_with_role_markers"]}</b></div>
    <div class="metric"><span>参与训练 label</span><b class="{train_ok}">{summary["trainable_labels"]}</b></div>
    <div class="metric"><span>存储的 labels</span><b>{summary["stored_labels"]}</b></div>
  </div>
  <p>实际加载 tokenizer：<code>{esc(result["tokenizer_source"])}</code></p>
  <p>页面只依赖 `parquet` 本身；会同时展示 parquet 原生 `labels[i]` 与按 `input_ids[i + 1]` 推导出的 label，便于检查它们是否一致。</p>
</section>

<section class="card">
  <h2>Parquet Schema</h2>
  <pre>{esc(parquet["schema"])}</pre>
</section>

<section class="card">
  <h2>Parquet 序列定位</h2>
  <pre>{esc(json.dumps({
      "parquet_rows": parquet["table_rows"],
      "sequences": [
          {
              "sequence_index": i,
              "row_index": seq["row_index"],
              "seq_index": seq["seq_index"],
              "seq_start": seq["seq_start"],
              "seq_end": seq["seq_end"],
              "packed_len": seq["packed_len"],
          }
          for i, seq in enumerate(parquet["sequences"])
      ],
  }, ensure_ascii=False, indent=2))}</pre>
</section>

<section class="card">
  <h2>所有 Parquet row 和子序列</h2>
  {''.join(sequence_sections)}
</section>
"""


def render_page(state: FormState, report: str = "", error: str = "") -> bytes:
    error_html = f'<section class="card"><h2 class="bad">错误</h2><pre>{esc(error)}</pre></section>' if error else ""
    body = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Packed SFT Parquet Viewer</title>
  {STYLE}
</head>
<body>
  <main>
    <h1>Packed SFT Parquet Viewer</h1>
    <p>输入 packed `parquet` 与对应 tokenizer，查看每个子序列的反解码文本、role 段落、token、loss mask 与训练 label。</p>
    {render_form(state)}
    {error_html}
    {report}
  </main>
</body>
</html>
"""
    return body.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.respond(render_page(FormState()))

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length).decode("utf-8")
        values = parse_qs(payload, keep_blank_values=True)
        state = form_from_values(values)
        try:
            result = analyze(state)
            self.respond(render_page(state, report=render_report(result)))
        except Exception:
            self.respond(render_page(state, error=traceback.format_exc()))

    def respond(self, content: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize packed Parquet SFT sequences.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving alignment viewer at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
