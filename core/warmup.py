"""引擎初始化预热：sampler 编译路径 + prefill eager 路径。

从 engine.py 抽出（行为不变）。Warmup 持有 engine 引用（预热需调用 engine 的
step/collect/update_sequences/get_next_batch 等完整推理循环），engine 在 __init__
末尾调用 warmup_sampler / warmup_prefill。
"""
import torch


class Warmup:
    """初始化预热器。消除首次调用的编译/算法选择开销（见各方法 docstring）。"""

    def __init__(self, engine):
        self.engine = engine

    def warmup_sampler(self, batch_sizes):
        """对所有捕获的 batch_size 预热 sampler 编译路径，消除首次调用的 ~1-2s 捕获开销。

        greedy（argmax）路径无编译开销；此处预热的是 temp>0 的 _compiled_sample 路径，
        覆盖连续批处理下任意 batch_size 首次采样。"""
        eng = self.engine
        vocab = eng.config.vocab_size
        dtype = next(eng.model.parameters()).dtype
        for bs in batch_sizes:
            fake_logits = torch.zeros(bs, vocab, dtype=dtype, device=eng.device)
            temps = torch.full((bs,), 0.01, device=eng.device)
            topp = torch.ones(bs, device=eng.device)
            rep = torch.ones(bs, device=eng.device)
            eng.sampler(fake_logits, temps, topp, 1000,
                        prev_tokens=None, rep_penalties=rep,
                        all_greedy=False, any_rep_pen=False)
        # 也预热 repetition-penalty 路径（bs=1 足矣，scatter 形状随 vocab 而靘认 prev_tokens 长度）
        fake_logits = torch.zeros(1, vocab, dtype=dtype, device=eng.device)
        prev = torch.tensor([[0, 1, 2]], dtype=torch.long, device=eng.device)
        rep = torch.tensor([1.1], device=eng.device)
        eng.sampler(fake_logits, torch.tensor([0.01], device=eng.device),
                    torch.ones(1, device=eng.device), 1000,
                    prev_tokens=prev, rep_penalties=rep,
                    all_greedy=False, any_rep_pen=True)

    def warmup_prefill(self, batch_sizes):
        """预热 prefill eager 路径，消除 cuBLAS/flash 首次跑每个 (B,S) shape 的算法选择开销。

        用极短 dummy prompt（~8 token）在代表性 batch size 跑一次 prefill+少量 decode，
        然后立即释放这些 dummy seq 的 KV block。代表 batch 取 cap_sizes 中 ≤32 的若干档
        （大 batch 算法选择与小 batch 同路径，无需每档都跑）。"""
        eng = self.engine
        from core.sequence import Sequence
        from core.inference_context import BatchInferenceContext
        # 选代表性 batch size：小 bs 密集测，大 bs 取一档（64）即可覆盖 cuBLAS 大矩阵路径。
        warm_sizes = [b for b in batch_sizes if b <= 32]
        if 64 in batch_sizes:
            warm_sizes.append(64)
        dummy_prompt = "warmup "  # ~2 token
        sid_base = 10_000_000  # 避免与真实 seq_id 冲突
        for bs in warm_sizes:
            dummy_seqs = []
            for i in range(bs):
                seq = Sequence(sid_base + i, dummy_prompt, eng.tokenizer, max_tokens=2)
                seq.temperature = 0.01; seq.top_p = 1.0
                if eng.eos_token_id is not None:
                    seq.eos_token_id = eng.eos_token_id
                eng.scheduler.add_request(seq)
                dummy_seqs.append(seq)
            # 跑 prefill + 1 decode（触发该 batch_size 的 prefill GEMM/flash + decode graph replay）
            for _ in range(20):
                b, bt = eng.get_next_batch()
                if not b:
                    break
                ctx = BatchInferenceContext(len(b), bt, b)
                eng.step(ctx); eng.collect(ctx); eng.update_sequences(ctx.sequences)
                if not eng.scheduler.running_sequences and not eng.scheduler.waiting_queue:
                    break
            # 清理 dummy seq 的 KV（释放 block）+ GDN 状态池 slot（幂等：已释放则 no-op）
            for seq in dummy_seqs:
                try:
                    eng.cache_manager.free(seq.seq_id)
                except Exception:
                    pass
                eng.adapter.on_seq_finished(seq)
            eng.scheduler.running_sequences.clear()
            eng.scheduler.finished_sequences.clear()
