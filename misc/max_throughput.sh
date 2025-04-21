python ./benchmarks/cpp/prepare_dataset.py \
  --stdout \
  --tokenizer nvidia/DeepSeek-R1-FP4 \
  token-norm-dist \
  --input-mean 10000 --output-mean 1000 \
  --input-stdev 0 --output-stdev 0 \
  --num-requests 16384 > dataset.txt

cat >./extra-llm-api-config.yml <<EOF
pytorch_backend_config:
  use_cuda_graph: true
  cuda_graph_padding_enabled: true
  enable_overlap_scheduler: true
  cuda_graph_batch_sizes:
  - 1
  - 2
  - 4
  - 8
  - 16
  - 32
  - 64
  - 128
enable_attention_dp: true
enable_chunked_prefill: true
kv_cache_config:
  free_gpu_memory_fraction: 0.6
EOF

trtllm-bench -m nvidia/DeepSeek-R1-FP4 \
  --model_path /dev/shm/models/r1-fp4 \
  throughput \
  --tp 8 \
  --ep 8 \
  --warmup 0 \
  --dataset ./dataset.txt \
  --backend pytorch \
  --max_batch_size 128 \
  --max_num_tokens 8192 \
  --num_requests 16384 \
  --concurrency 1024 \
  --kv_cache_free_gpu_mem_fraction 0.6 \
  --extra_llm_api_options ./extra-llm-api-config.yml