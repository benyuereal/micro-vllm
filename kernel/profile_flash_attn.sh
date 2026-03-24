#!/bin/bash
# ---------------------------------------------------------------------------
# profile_flash_attn.sh — NCU profiling for triton_flash_attn kernels
#
# Usage:
#   cd /root/micro-vllm
#   bash kernel/profile_flash_attn.sh [decode|prefill|all]
#
# Output: ncu_flash_attn.ncu-rep  (open with NVIDIA Nsight Compute GUI)
# ---------------------------------------------------------------------------

MODE=${1:-all}   # decode | prefill | all

# ---------------------------------------------------------------------------
# Metrics legend
#   gpu__dram_throughput        → DRAM BW utilisation (% of peak)
#   sm__throughput              → SM compute utilisation (% of peak)
#   l1tex__t_sector_hit_rate    → L1 cache hit rate (higher = more reuse)
#   sm__warps_active            → warp occupancy (% of peak)
#   stalled_long_scoreboard     → % warps stalled waiting for global mem
#   stalled_mio_throttle        → % warps stalled on mem-issue queue full
#   stalled_math_pipe_throttle  → % warps stalled waiting for math pipes
#   smsp__inst_executed         → total instructions per warp (IPC proxy)
# ---------------------------------------------------------------------------

METRICS=(
  "gpu__time_duration.sum"
  "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
  "l1tex__t_sector_hit_rate.pct"
  "sm__throughput.avg.pct_of_peak_sustained_elapsed"
  "sm__warps_active.avg.pct_of_peak_sustained_active"
  "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"
  "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct"
  "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct"
  "smsp__inst_executed.avg.per_cycle_active"
)

# Build comma-separated metric string
METRIC_STR=$(IFS=,; echo "${METRICS[*]}")

# Choose which kernel names to capture
case "$MODE" in
  decode)
    FILTER="--kernel-name regex:_flash_decode_kernel"
    ;;
  prefill)
    FILTER="--kernel-name regex:_flash_prefill_kernel"
    ;;
  *)
    FILTER=""
    ;;
esac

echo "=== NCU Flash Attention profiling (mode=$MODE) ==="
echo "Metrics: ${METRICS[*]}"
echo ""

ncu \
  --profile-from-start off \
  --metrics "$METRIC_STR" \
  $FILTER \
  --target-processes all \
  --export ncu_flash_attn \
  --force-overwrite \
  python kernel/profile_flash_attn.py

echo ""
echo "Report written to ncu_flash_attn.ncu-rep"
echo "Open with: ncu-ui ncu_flash_attn.ncu-rep"
echo ""
echo "=== Quick bottleneck guide ==="
echo "  DRAM > 80%, SM < 30%   → memory bandwidth bound (typical for decode GEMV)"
echo "  SM   > 60%, DRAM < 50% → compute bound          (typical for prefill GEMM)"
echo "  Both < 40%             → latency / occupancy bound"
echo "    → check: warps_active (low?), stalled_long_scoreboard (high?)"
echo "  stalled_long_scoreboard > 50% → global-mem latency dominant"
echo "    → solutions: larger BLOCK_N (more KV per iteration), num_stages pipeling"
echo "  stalled_mio_throttle > 30%   → too many in-flight mem ops"
echo "    → solutions: reduce BLOCK_N, fewer outstanding loads per warp"
