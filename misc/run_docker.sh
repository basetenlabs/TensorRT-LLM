docker run -d --gpus all \
  -v /dev/shm:/dev/shm \
  -v $(pwd):/workspace/TensorRT-LLM \
  --network host \
  -p 8000:8000 \
  baseten/tensorrt_llm-release:v0.20.0rc1 \
  tail -f /dev/null
