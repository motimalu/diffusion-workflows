# see: https://huggingface.co/Qwen/Qwen3.6-27B
# We recommend using the following set of sampling parameters for generation

# Thinking mode for general tasks:
# temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0
# Thinking mode for precise coding tasks (e.g. WebDev):
# temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0
# Instruct (or non-thinking) mode:
# temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
# Please note that the support for sampling parameters varies according to inference frameworks.

python bulk-caption-extended.py \
  --model-name="Qwen3.6-27B" \
  --max-tokens=81920 \
  --temperature=0.6 \
  --top-p=0.95 \
  --top-k=20 \
  --presence-penalty=0.0 \
  --repetition-penalty=1.1 \
  --enable-thinking
