# Installation

## vLLM server

```
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

Serve with either:

```
./serve-moe-35b.sh
./serve-moe-122b.sh
```

## OpenAI inference script

```
uv pip install -r requirements.txt
```

Run inference with:

```
python bulk-caption-extended.py
```
