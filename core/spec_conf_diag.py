"""target 置信度诊断（投机解码 B/A 分桶）。

把"draft 错"的步分成 B 桶（target 很确定但 draft 猜错，draft 可救）与 A 桶
（target 自己不确定，救不了）。每个 draft 位置 p 记录：
    target_conf[p] = softmax(verify_logits[p])[target_argmax[p]]
只统计到本步第一个 mismatch 位置为止（p <= accepted）：之后位置的 target 预测
条件在错误前缀上，"正确 token"定义变了，conf 无意义。

按上下文长度分档（short<512 / mid 512-1500 / long>=1500，ctx_len=kv_len-1）+ all
汇总。由 MICRO_SPEC_TGT_CONF=1 开启（默认关，SpecEngine 不构造本对象 → 零开销）。
"""
import torch


class TargetConfDiag:
    def __init__(self, N: int):
        self.N = N
        self.stats = self._blank_all()

    def _blank(self):
        N = self.N
        return {
            "steps": 0,
            "valid": [0] * N,          # p 有效（p<=accepted）的步数
            "match": [0] * N,          # draft[p]==target[p] 的步数
            "misB5": [0] * N, "misA5": [0] * N,   # mismatch 且 conf>=0.5 / <0.5
            "misB9": [0] * N, "misA9": [0] * N,   # mismatch 且 conf>=0.9 / <0.9
            "conf": [[] for _ in range(N)],        # 有效位置 conf 全量（分布用）
            "conf_mis": [[] for _ in range(N)],    # mismatch 位置 conf（分布用）
        }

    def _blank_all(self):
        return {"all": self._blank(), "short": self._blank(),
                "mid": self._blank(), "long": self._blank()}

    def reset(self):
        self.stats = self._blank_all()

    @staticmethod
    def bucket(ctx_len: int) -> str:
        if ctx_len < 512:
            return "short"
        if ctx_len < 1500:
            return "mid"
        return "long"

    def record(self, vlogits, target_preds, d_cpu, t_cpu, accepted, kv_len):
        """每步调用：统计到本步第一个 mismatch 位置为止（p <= accepted）。
        conf[p] = target 在位置 p 对自己 greedy token 的 softmax 概率（fp32，
        vocab 248k 下 8 行 softmax 临时 ~8MB，可忽略）。"""
        b = self.bucket(kv_len - 1)
        for s in (self.stats["all"], self.stats[b]):
            s["steps"] += 1
        conf = torch.softmax(vlogits.float(), dim=-1)
        conf = conf.gather(1, target_preds.unsqueeze(1)).squeeze(1)
        conf_cpu = conf.cpu().tolist()
        # draft 位置只有 0..N-1（p=N 是 bonus，非 draft 提议）。
        # accepted<N 时有效到 accepted（首个 mismatch）；accepted==N 时到 N-1。
        for p in range(min(accepted + 1, self.N)):
            for s in (self.stats["all"], self.stats[b]):
                s["valid"][p] += 1
                s["conf"][p].append(conf_cpu[p])
                if d_cpu[p] == t_cpu[p]:
                    s["match"][p] += 1
                else:
                    c = conf_cpu[p]
                    s["conf_mis"][p].append(c)
                    if c >= 0.5:
                        s["misB5"][p] += 1
                    else:
                        s["misA5"][p] += 1
                    if c >= 0.9:
                        s["misB9"][p] += 1
                    else:
                        s["misA9"][p] += 1

    def report(self):
        """每位置 p：valid/match 率、mismatch 的 B/A 分桶（阈值 0.5 与 0.9）、
        conf 分布（均值/分位数）。"""
        out = {}
        for name, st in self.stats.items():
            nb = st["steps"]
            if nb == 0:
                out[name] = {"steps": 0}
                continue
            pos = {}
            for p in range(self.N):
                nv = st["valid"][p]
                if nv == 0:
                    pos[str(p)] = {"valid": 0}
                    continue
                confs = sorted(st["conf"][p])

                def _q(f, _c=confs):
                    return _c[min(len(_c) - 1, int(f * len(_c)))]

                cm = st["conf_mis"][p]
                pos[str(p)] = {
                    "valid": nv,
                    "match_rate": st["match"][p] / nv,
                    "mis": nv - st["match"][p],
                    "B5": st["misB5"][p], "A5": st["misA5"][p],
                    "B9": st["misB9"][p], "A9": st["misA9"][p],
                    "conf_mean": sum(confs) / len(confs),
                    "conf_p10": _q(0.1), "conf_p50": _q(0.5), "conf_p90": _q(0.9),
                    "conf_mis_mean": (sum(cm) / len(cm)) if cm else None,
                }
            out[name] = {"steps": nb, "pos": pos}
        return out
