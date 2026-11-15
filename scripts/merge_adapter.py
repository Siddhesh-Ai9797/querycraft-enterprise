"""
Merges the QueryCraft LoRA adapter into Phi-3 Mini and saves one plain model.

vLLM and llama.cpp both need a single merged model instead of base plus adapter.

    python scripts/merge_adapter.py                 # saves to models/querycraft-phi3-merged
    python scripts/merge_adapter.py --out /models/x # any folder
"""
import argparse
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")

BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_ID    = "Sid9797/querycraft-phi3-sql"


def merge(out_dir: str, device: str = "auto") -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {BASE_MODEL_ID} in float16...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, device_map=device, trust_remote_code=False,
    )
    print(f"Applying adapter {ADAPTER_ID}...")
    model = PeftModel.from_pretrained(base, ADAPTER_ID)
    model = model.merge_and_unload()
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE_MODEL_ID).save_pretrained(out_dir)
    print(f"✅ Merged model saved to {out_dir}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models/querycraft-phi3-merged")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda:0")
    args = parser.parse_args()
    merge(args.out, args.device)
