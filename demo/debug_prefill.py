"""Debug: 对比 HF vs micro 的 prefill logits / 逐层 hidden，定位 Qwen3.5 prefill bug。"""
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ.get("MODEL_PATH", "/models/Qwen3.5-0.8B")
PROMPT = "The capital of France is"


def hf_forward(prompt):
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, local_files_only=True)
    model.eval()
    ids = tok.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([ids], device="cuda:0")
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
    # hidden_states[0]=embed, [1..n]=每层输出, 共 n+1 个（未过 final norm）
    layer_hidden = [hs[0] for hs in out.hidden_states]
    logits = out.logits[0, -1].float()
    return ids, logits, layer_hidden, model


def micro_forward(prompt):
    from core.engine import InferenceEngine
    from core.inference_context import BatchInferenceContext
    from models.base import PrefillMeta
    eng = InferenceEngine(MODEL, max_batch_size=8, max_prefill_tokens=4096)
    tok = eng.tokenizer
    ids = tok.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor(ids, device=eng.device)
    # 手动构造 prefill meta（单 seq，整条）
    L = len(ids)
    cm = eng.cache_manager
    bs = cm.block_size
    ok, _ = cm.alloc(ids[0] if False else 12345, L)  # 用固定 seq_id
    # 重新用真实 seq_id
    cm.free(12345)
    sid = 99999
    ok, _ = cm.alloc(sid, L)
    cu_q = torch.tensor([0, L], dtype=torch.int32, device=eng.device)
    cu_k = torch.tensor([0, L], dtype=torch.int32, device=eng.device)
    pos = torch.arange(L, device=eng.device)
    bt = cm._block_table_buffer[:1]
    cm.cache_batch_data([sid], [L])
    abs_pos = torch.arange(L, device=eng.device)
    slot = (bt[0, abs_pos // bs] * bs + abs_pos % bs).to(torch.int32)
    meta = PrefillMeta(cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, position_ids=pos,
                       slot_mapping=slot, block_table=bt, n_seqs=1,
                       max_seqlen_q=L, max_seqlen_k=L)
    # 清零 GDN 状态池（warmup 残留会污染 GDN 层；真实路径 on_prefill_batch 会清零）
    eng.prefill_runner._gdn_state_pool.zero_()
    eng.prefill_runner._gdn_conv_state_pool.zero_()
    # 逐层 hidden
    text_model = eng.adapter._text_model(eng.model)
    with torch.no_grad():
        h = text_model.embed_tokens(input_ids)  # [L, H]
        layer_hidden = [h]
        for i, layer in enumerate(text_model.layers):
            h = eng.prefill_runner.adapter.prefill(layer, h, i, eng.prefill_runner, cm, meta)
            layer_hidden.append(h)
        h = text_model.norm(h)
        logits = eng.model.lm_head(h)
    return ids, logits[-1].float(), layer_hidden, eng


def main():
    hf_ids, hf_logits, hf_hidden, hf_model = hf_forward(PROMPT)
    print(f"HF ids: {hf_ids}")
    print(f"HF last-token logits top5: {hf_logits.topk(5)}")

    micro_ids, micro_logits, micro_hidden, eng = micro_forward(PROMPT)
    print(f"micro ids: {micro_ids}")
    print(f"micro last-token logits top5: {micro_logits.topk(5)}")

    # logits 对比
    diff = (hf_logits - micro_logits).abs()
    print(f"\nlogits: max_abs_diff={diff.max().item():.4f} mean={diff.mean().item():.6f}")
    print(f"  argmax hf={hf_logits.argmax().item()} micro={micro_logits.argmax().item()}")

    print("\nshapes: hf_hidden[0]", tuple(hf_hidden[0].shape),
          "micro_hidden[0]", tuple(micro_hidden[0].shape))
    print("hf_hidden[24]", tuple(hf_hidden[24].shape),
          "micro_hidden[24]", tuple(micro_hidden[24].shape))
    # 逐层 hidden 对比（取最后一个 token）
    print("\n逐层 hidden 对比（last token）:")
    print(f"  {'layer':>5} {'hf_norm':>10} {'micro_norm':>10} {'max_diff':>10} {'cos':>10}")
    for i in range(len(hf_hidden)):
        a = hf_hidden[i][-1].float().flatten()   # HF 已 squeeze 掉 batch 维
        b = micro_hidden[i][-1].float().flatten()
        d = (a - b).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        flag = "  <-- DIVERGE" if d > 0.5 else ""
        print(f"  {i:5d} {a.norm().item():10.4f} {b.norm().item():10.4f} {d:10.4f} {cos:10.6f}{flag}")
    # 也看第一个 token（embedding 应完全一致）
    a0 = hf_hidden[0][0].float().flatten()
    b0 = micro_hidden[0][0].float().flatten()
    print(f"\n  first-token embed: hf_norm={a0.norm().item():.4f} micro_norm={b0.norm().item():.4f} "
          f"max_diff={(a0-b0).abs().max().item():.6f}")
    print(f"  hf embed[0,:4]={a0[:4].tolist()}")
    print(f"  micro embed[0,:4]={b0[:4].tolist()}")


if __name__ == "__main__":
    main()
