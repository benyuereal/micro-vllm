"""introspect HF cache 结构，找 GDN recurrent/conv state 的存放位置。"""
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3.5-0.8B")
PROMPT = "The capital of France is"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, local_files_only=True)
    model.eval()
    ids = tok.encode(PROMPT, add_special_tokens=True)
    cur = torch.tensor([ids], device="cuda:0")
    with torch.no_grad():
        out = model(cur, use_cache=True)
    cache = out.past_key_values
    print("cache type:", type(cache))
    print("cache attrs:", [a for a in dir(cache) if not a.startswith("__")])
    try:
        print("len(cache):", len(cache.layers))
        layer0 = cache.layers[0]
        print("layer0 type:", type(layer0))
        print("layer0 attrs:", [a for a in dir(layer0) if not a.startswith("__")])
        for a in ("recurrent_states", "conv_states"):
            if hasattr(layer0, a):
                v = getattr(layer0, a)
                print(f"  layer0.{a}: type={type(v)} keys={list(v.keys()) if isinstance(v,dict) else 'n/a'}")
                if isinstance(v, dict):
                    for k, val in v.items():
                        print(f"    key={k} type={type(val)} shape={getattr(val,'shape',None)} dtype={getattr(val,'dtype',None)}")
    except Exception as e:
        print("introspect err:", e)


if __name__ == "__main__":
    main()
