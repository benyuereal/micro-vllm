"""对比 micro vs HF 的 final hidden state（post final_norm, pre lm_head）@ pos4，
判断 logit 差异来自 backbone 还是 lm_head GEMV。"""
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3.5-0.8B")
PROMPT = "中国的首都是"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, local_files_only=True)
    hf.eval()
    ids = tok.encode(PROMPT, add_special_tokens=True)
    text = hf.model

    # HF: 前缀 [ids, 98116, 3709] → 取 pos4 的 final hidden（pre lm_head）
    prefix = ids + [98116, 3709]
    cur = torch.tensor([prefix], device="cuda:0")
    hhook = {}
    def hook(mod, inp, out):
        hhook["h"] = out.detach().clone()
    # final norm 是 text.norm
    h = text.norm
    hh = h.register_forward_hook(hook)
    with torch.no_grad():
        out = hf(cur)
    hh.remove()
    hf_hidden = hhook["h"][0, -1].float()  # [1024] pos4
    hf_logits = out.logits[0, -1].float()

    # micro: 跑 prefill + 2 decode，抓 pos4 的 hidden
    from core.engine import InferenceEngine
    from core.inference_context import BatchInferenceContext
    eng = InferenceEngine(MODEL, max_batch_size=64, max_prefill_tokens=4096)
    eng.add_request(PROMPT, 8, temperature=0.0, top_p=1.0)
    # hook final_norm：graph_runner 里 final norm 后是 lm_head
    # 用 adapter 的 final_norm 模块 hook
    mhook = {}
    fn = eng.graph_runner.adapter.final_norm(eng.model)
    def mhook_fn(mod, inp, out):
        mhook["h"] = out.detach().clone()
    mh = fn.register_forward_hook(mhook_fn)
    micro_step_logits = []
    while True:
        b, bt = eng.get_next_batch()
        if not b:
            break
        ctx = BatchInferenceContext(len(b), bt, b)
        eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
        if bt == "decode":
            micro_step_logits.append(ctx.logits[0].float().clone())
    mh.remove()
    # decode step 1 = pos4
    micro_hidden = mhook["h"][-1].float()  # 最后一次 forward 的 hidden
    micro_logits = micro_step_logits[1]

    d = (hf_hidden - micro_hidden).abs()
    cos = F.cosine_similarity(hf_hidden, micro_hidden, dim=0).item()
    print("final hidden @pos4: max_diff=%.5f cos=%.6f" % (d.max().item(), cos))
    print("  hf_norm=%.4f micro_norm=%.4f" % (hf_hidden.norm().item(), micro_hidden.norm().item()))
    print("  hf mean=%.4f micro mean=%.4f" % (hf_hidden.mean().item(), micro_hidden.mean().item()))
    # 用 HF 的 lm_head 权重算 micro hidden 的 logits，看 GEMV 是否引入差异
    lm_w = hf.lm_head.weight.float()  # [vocab, 1024]
    logits_from_hf_w = micro_hidden @ lm_w.t()
    d2 = (logits_from_hf_w - hf_logits).abs()
    print("micro_hidden @ HF_lmW vs HF_logits: max_diff=%.5f" % d2.max().item())
    # micro 自己的 logits vs 用 HF 权重算的
    d3 = (micro_logits - logits_from_hf_w).abs()
    print("micro_logits vs (micro_hidden@HF_lmW): max_diff=%.5f" % d3.max().item())
    # 用 micro hidden + micro 的 lm_head（tied embed）
    emb = eng.graph_runner.adapter.embed(eng.model).weight.float()
    logits_emb = micro_hidden @ emb.t()
    d4 = (micro_logits - logits_emb).abs()
    print("micro_logits vs (micro_hidden@micro_emb): max_diff=%.5f" % d4.max().item())


if __name__ == "__main__":
    main()
