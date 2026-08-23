"""HTTP 并发 /generate 压测：N 条独立 max_tokens 的并发请求打 micro-vllm API 服务。

测什么：
  - 通过 HTTP 并发调用 micro-vllm 服务的 /generate 接口
  - N 条请求各自随机 max_tokens（100 ~ MAX_OUT），统计聚合吞吐 tok/s
  - 与 nano 进程内批处理（bench_nano_batch.py）同 seed 同分布，口径对齐

用法：
  python3 bench_batch_compare.py <url> <N>
  例: python3 bench_batch_compare.py http://localhost:8000 256
  环境变量 MAX_OUT 控制 max_tokens 上限（默认 1024）

依赖：
  - 已启动的 micro-vllm API 服务（python api_server.py，GPU 由服务进程指定）
  - requests 库
"""
import os, sys, time, requests, concurrent.futures
from random import randint, seed

URL = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 256
MAX_OUT = int(os.environ.get("MAX_OUT", "1024"))

seed(0)
# 不同 prompt 避免 dict 合并；input 长度对 decode 吞吐影响小，用短 prompt
prompts = [f"Benchmark {i} " for i in range(N)]
max_tokens_list = [randint(100, MAX_OUT) for _ in range(N)]
total_out = sum(max_tokens_list)


def call(i):
    r = requests.post(URL + "/generate", json={"prompt": prompts[i], "max_tokens": max_tokens_list[i],
                                               "temperature": 0.01, "top_p": 1.0}, timeout=600)
    return len(r.json().get("text", ""))


t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(N) as ex:
    res = list(ex.map(call, range(N)))
t = time.time() - t0
ok = sum(1 for x in res if x > 0)
print(f"N={N} ok={ok}/{N} total_out_tok={total_out} time={t:.2f}s "
      f"throughput={total_out/t:.1f} tok/s")
