export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export SAFETENSORS_FAST_GPU=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=12
export VLLM_SLEEP_WHEN_IDLE=1
export VLLM_DISABLE_PYNCCL=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_NVFP4_GEMM_BACKEND=cutlass
export VLLM_USE_FLASHINFER_MOE_FP4=0

python -m vllm.entrypoints.openai.api_server \
  --model Sehyo/Qwen3.5-122B-A10B-NVFP4 \
  --served-model-name "Qwen3.5-122B-A10B" \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-prefix-caching \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len auto \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 1 \
  --reasoning-parser qwen3 \
  --enable-chunked-prefill \
  --allowed-local-media-path /mnt/c/data/ 