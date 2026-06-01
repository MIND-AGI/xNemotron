# Packed SFT Parquet Viewer

本工具现在只查看导出的 `shard_000000.parquet`，不再依赖 `jsonl`。它适合直接检查 packed 数据中的：

- `input_ids`
- `loss_mask`
- `labels`
- `seq_start_id`
- 每个子序列的反解码文本
- 按 `<|im_start|>role` 切分后的 role 段落
- 每个位置真正参与训练的 next-token `label`

## 查看逻辑

- 先读取 parquet 的 `input_ids`、`loss_mask`、`labels`、`seq_start_id`。
- 使用 `seq_start_id` 把每个 packed row 切成多个原始 sequence。
- 用 tokenizer 反解码每个 sequence。
- 如果文本中存在 `<|im_start|>role`，就按 role 段落切分展示；否则整段标记为 `unknown`。
- token 表中使用：
  - `labels[i]`：parquet 中实际存储的监督标签；通常未训练位置为 `-100`
  - `label_id[i] = input_ids[i + 1]`
  - `loss_mask[i] == 1` 时，该 label 参与训练 loss。

## 启动

```bash
cd /Users/bytedance/Movies/btdr/Nemotron

uv run --extra xenna python visualization/packed_sft_alignment_viewer.py \
  --host 127.0.0.1 \
  --port 8765
```

然后打开：

```text
http://127.0.0.1:8765
```

## 页面输入

- `Parquet 文件路径`：导出的 `shard_000000.parquet`。
- `Tokenizer`：训练数据处理时使用的 tokenizer 名称、本地 snapshot 路径，或 HF cache 根目录。
- `HF_HOME`：可选，自定义 HuggingFace cache 路径。
- 每个 parquet 子序列都会完整展示，不再按 token 行数截断。

## 判断标准

- `参与训练 label` 应大于 `0`。
- `存储的 labels` 可以直接看到 parquet 里非 `-100` 的监督位置数。
- `带 role marker 的序列` 越多，说明反解码文本里越多保留了 `<|im_start|>role` 结构。
- 每个 sequence 卡片中的 role 段落表会展示 role、文本、token ids、loss mask、labels 和 `seq_start_id`。
- 每个 sequence 卡片也包含逐 token 表；该表按连续 role 折叠，方便检查 token、mask、`labels[i]` 与 `input_ids[i + 1]` 的关系。

如果 role 段落异常，通常是：

- tokenizer 与生成 parquet 时不一致；
- parquet 中本身没有保留 `<|im_start|>role` 标记；
- sequence 被裁切后只剩尾部文本，因此会看到 `tail` 或 `unknown`。

## Tokenizer 路径说明

如果填 repo id，例如：

```text
nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

需要该 repo 已经在当前 `HF_HOME` 或默认 HuggingFace cache 中完整缓存。

如果填 HF hub cache 根目录，例如：

```text
/Users/bytedance/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

工具会自动解析到：

```text
/Users/bytedance/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/<revision>
```

也可以直接填 snapshot 目录。注意 tokenizer 必须和生成 `shard_000000.parquet` 时使用的 tokenizer 完全一致，否则反解码文本和 role 切分都会失真。
