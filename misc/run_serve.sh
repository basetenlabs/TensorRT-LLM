docker run -it --gpus all \
  -v /dev/shm:/dev/shm \
  -v ./extra-llm-api-config.yml:/extra-llm-api-config.yml \
  -e TRTLLM_ENABLE_PDL=1 \
  --network host \
  -p 8000:8000 \
  baseten/tensorrt_llm-release:0.19.0rc0 \
  trtllm-serve serve \
    --tp_size 8 \
    --ep_size 4 \
    --backend pytorch \
    --extra_llm_api_options /extra-llm-api-config.yml \
    /dev/shm/models/r1-fp4/
