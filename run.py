import os
import torch
from tensorrt_llm._torch import LLM
from transformers import AutoTokenizer
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig


def main() -> None:
    
    pytorch_config = PyTorchConfig(
        attn_backend="FLASHINFER",
    )

    llm = LLM(
        model="/app/tensorrt_llm/TensorRT-LLM/scout",
        # model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        backend="flashinfer",
        tensor_parallel_size=4,
        moe_tensor_parallel_size=4,
        moe_expert_parallel_size=1,
        max_seq_len=16192,
        max_input_len=12000,
        max_num_tokens=16192,
        pytorch_backend_config=pytorch_config,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "/app/tensorrt_llm/TensorRT-LLM/scout",
    )

    # with open("./long_context_code.py", "r") as f:
    with open("./long_text.txt", "r") as f:
        text = f.read()
    
    # Tokenize the text
    tokens = tokenizer(text, return_tensors="pt")
    prompt = tokenizer.decode(tokens["input_ids"][0][:16000], skip_special_tokens=True)

    out = llm.generate(
        prompt,
    )
    
    print(out)

if __name__ == "__main__":
    main()