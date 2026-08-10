#!/usr/bin/env python3
"""端到端吞吐对比：baseline (Triton MoE) vs TileLang MoE。

启动 server 后运行：python3 bench_moe_e2e.py
"""
import sys
import time
import requests
from transformers import AutoTokenizer

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"
MODEL = "/models/DeepSeek-V2-Lite"
tok = AutoTokenizer.from_pretrained(MODEL)

PROMPTS = [
    "写一个 SpringBoot 文件上传的完整代码示例，包含前端和后端",
    "详细解释量子力学中的叠加态和纠缠现象，并举例说明",
    "用 Python 实现快速排序和归并排序，并分析时间复杂度",
    "介绍宋朝的历史，包括北宋和南宋的重要事件和人物",
]

def bench_one(prompt, max_tokens=200):
    t0 = time.perf_counter()
    r = requests.post(f"{URL}/generate", json={
        "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.7
    }).json()
    t1 = time.perf_counter()
    text = r.get("text", "")
    gen = text[len(prompt):] if text.startswith(prompt) else text
    gen_ids = tok.encode(gen, add_special_tokens=False)
    wall = t1 - t0
    return len(gen_ids), wall, len(gen_ids) / wall if wall > 0 else 0

def main():
    print(f"Testing {URL}")
    # warmup
    bench_one(PROMPTS[0], max_tokens=20)
    time.sleep(0.5)

    all_tps = []
    for pi, p in enumerate(PROMPTS):
        for run in range(3):
            n, wall, tps = bench_one(p, max_tokens=200)
            tag = "" if run > 0 else " (warmup)"
            print(f"  p{pi} run{run}{tag}: {n} tok / {wall:.2f}s = {tps:.1f} tok/s")
            if run > 0:
                all_tps.append(tps)
            time.sleep(0.2)

    print(f"\n=== 结果 ===")
    print(f"  mean (排除warmup): {sum(all_tps)/len(all_tps):.1f} tok/s")
    print(f"  median: {sorted(all_tps)[len(all_tps)//2]:.1f} tok/s")
    print(f"  min: {min(all_tps):.1f}  max: {max(all_tps):.1f}")

if __name__ == "__main__":
    main()
