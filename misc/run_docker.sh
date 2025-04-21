docker run -it --gpus all \
  -it \
  -v /dev/shm:/dev/shm \
  -v ./extra-llm-api-config.yml:/extra-llm-api-config.yml \
  --network host \
  -p 8000:8000 \
  baseten/tensorrt_llm-release:0.19.0rc0 \
  /bin/bash
