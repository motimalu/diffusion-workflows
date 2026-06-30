# see: https://huggingface.co/Qwen/Qwen3.5-122B-A10B
# Best Practices
# To achieve optimal performance, we recommend the following settings:

# Sampling Parameters:

# We suggest using the following sets of sampling parameters depending on the mode and task type:
# Thinking mode for general tasks:
# temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
# Thinking mode for precise coding tasks (e.g., WebDev):
# temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0
# Instruct (or non-thinking) mode for general tasks:
# temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
# Instruct (or non-thinking) mode for reasoning tasks:
# temperature=1.0, top_p=1.0, top_k=40, min_p=0.0, presence_penalty=2.0, repetition_penalty=1.0
# For supported frameworks, you can adjust the presence_penalty parameter between 0 and 2 to reduce endless repetitions. However, using a higher value may occasionally result in language mixing and a slight decrease in model performance.


python bulk-caption-extended.py \
  --model-name="Qwen3.5-122B-A10B" \
  --max-tokens=81920 \
  --temperature=0.6 \
  --top-p=0.95 \
  --top-k=20 \
  --presence-penalty=0.0 \
  --repetition-penalty=1.0 \
  --enable-thinking
