"""Compare locally measured throughput with OpenRouter reported throughput."""
from openai import OpenAI
import time
import requests
import numpy as np

OR_API_KEY = ""
OR_META_URL = "https://openrouter.ai/api/v1/generation"
OR_URL = "https://openrouter.ai/api/v1"
OR_HEADERS = {"Authorization": f"Bearer {OR_API_KEY}"}

client = OpenAI(
  base_url=OR_URL,
  api_key=OR_API_KEY,
)
stream = client.chat.completions.create(
  model="deepseek/deepseek-r1",
  messages=[
    {
      "role": "user",
      "content": "Hi there"
    }
  ],
  stream=True,
  stream_options={"include_usage": True},
  max_tokens=1024,
  extra_body={
    "provider": {
        "order": ["DeepInfra"],
        "quantizations": ["fp4"],
    }
  }
)

T = []
decode_start = None
reasoning_end = None
context_start = time.time()
elem = None
for elem in stream:
  curr_time = time.time()
  T.append(curr_time)
  if decode_start is None:
      decode_start = curr_time
  if reasoning_end is None and elem.choices[0].delta.content:
    reasoning_end = curr_time
decode_end = time.time()

assert decode_start is not None
assert elem is not None
assert elem.usage is not None

assert len(T) > 1
D = np.diff(T)
med_itl = np.median(D)
avg_itl = np.mean(D)
std_itl = np.std(D)
min_itl = np.min(D)
max_itl = np.max(D)

prefill = decode_start - context_start
reasoning = reasoning_end - decode_start if reasoning_end else None
generation = decode_end - decode_start
total = decode_end - context_start
completion_tokens = elem.usage.completion_tokens
tps = completion_tokens / generation

print("LOCALLY OBSERVED METRICS")
print("prefill", prefill, "s")
print("reasoning", reasoning, "s")
print("generation", generation, "s")
print("total", total, "s")
print("completion tokens", completion_tokens, "tokens")
print("tps", tps, "tps")
print()
print("med_itl", med_itl)
print("avg_itl", avg_itl)
print("std_itl", std_itl)
print("min_itl", min_itl)
print("max_itl", max_itl)
print()

time.sleep(1)  # Metadata API needs some time to be ready.

metadata_response = requests.get(OR_META_URL, headers=OR_HEADERS, params={"id": elem.id})
metadata = metadata_response.json()

prefill = metadata["data"]["latency"] / 1000
generation = metadata["data"]["generation_time"] / 1000
total = prefill + generation
completion_tokens = metadata["data"]["tokens_completion"]
tps = completion_tokens / generation


print("OR REPORTED METRICS")
print("prefill", prefill, "s")
print("generation", generation, "s")
print("total", total, "s")
print("completion tokens", completion_tokens, "tokens")
print("tps", tps, "tps")
print()
