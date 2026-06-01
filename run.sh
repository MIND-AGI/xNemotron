WANDB_MODE=disabled \                                                     
uv run --extra xenna python src/nemotron/recipes/nano3/stage1_sft/data_prep.py \
  --config src/nemotron/recipes/nano3/stage1_sft/config/data_prep/xsft.yaml \
  blend_path=/Users/bytedance/Movies/automodel/Nemotron/src/nemotron/recipes/nano3/stage1_sft/config/data_prep/custom_blend.json \
  output_dir=/Users/bytedance/Movies/automodel/Nemotron/datasets/smoke_sft_packed \
  chat_template=/Users/bytedance/Movies/automodel/Nemotron/src/nemotron/data_prep/templates/nano3_interleaved.jinja \
  sample=32 \
  num_shards=1 \
  train_ratio=1.0 \
  valid_ratio=0.0 \
  test_ratio=0.0 \
  used_in_filter=null \
  execution_mode=batch \
  force=true

uv run --extra xenna python visualization/packed_sft_alignment_viewer.py \
  --host 127.0.0.1 \
  --port 8765