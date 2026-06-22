export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export SAFETENSORS_FAST_GPU=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=12
export VLLM_SLEEP_WHEN_IDLE=1
export VLLM_DISABLE_PYNCCL=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.6-35B-A3B-FP8 \
    --served-model-name "Qwen3.6-35B-A3B" \
    --port 8000 \
    --enable-prefix-caching \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 262144 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 128 \
    --reasoning-parser qwen3 \
    --allowed-local-media-path /mnt/c/data/