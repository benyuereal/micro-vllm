import torch
import time
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.layer.std_mlp import MLP

try:
    import cpp_mlp
    HAS_CPP = True
    print("✅ C++ Extension (cpp_mlp) loaded")
except ImportError as e:
    HAS_CPP = False
    print(f"❌ Failed to load: {e}")
    exit(1)

def make_data(batch_size=1, num_heads=32, head_size=128, hidden_size=4096, intermediate_size=11008, device='cuda', dtype=torch.float16):
    """
    生成测试数据 - 使用保守的小值初始化避免 fp16 溢出
    """
    # 关键：使用小的随机值，避免 fp16 溢出
    # 真实模型权重通常是正态分布 N(0, 0.02) 级别
    
    return {
        'hidden': torch.randn(batch_size, 1, hidden_size, device=device, dtype=dtype) * 0.01,
        'attn_out': torch.randn(batch_size, num_heads, head_size, device=device, dtype=dtype) * 0.01,
        # 权重使用 Xavier 初始化范围
        'attn_proj_weight': torch.randn(hidden_size, hidden_size, device=device, dtype=dtype) * 0.02,
        'norm_weight': torch.ones(hidden_size, device=device, dtype=dtype),  # RMSNorm weight 通常是 1
        'gate_up_weight': torch.randn(hidden_size, 2*intermediate_size, device=device, dtype=dtype) * 0.02,
        'down_weight': torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype) * 0.02,
        'eps': 1e-6
    }

def check_tensor(t, name):
    """检查张量是否包含 nan/inf"""
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    if has_nan or has_inf:
        print(f"  ⚠️  {name}: min={t.min():.2e}, max={t.max():.2e}, has_nan={has_nan}, has_inf={has_inf}")
        return False
    return True

def test_correct():
    """功能测试"""
    print("\n🧪 功能测试 (数值正确性)")
    print("-" * 50)
    
    # 使用保守初始化
    d = make_data()
    
    # 检查输入
    print("Checking inputs...")
    all_valid = True
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            all_valid &= check_tensor(v, k)
    if not all_valid:
        print("❌ Input contains nan/inf, adjust initialization")
    
    # Python 版本
    print("Running Python MLP...")
    try:
        py_out = MLP.forward(**d)
        check_tensor(py_out, "Python output")
    except Exception as e:
        print(f"❌ Python error: {e}")
        return False
    
    # C++ 版本
    print("Running C++ MLP...")
    try:
        cpp_out = cpp_mlp.forward(
            d['hidden'], d['attn_out'], 
            d['attn_proj_weight'], d['norm_weight'],
            d['gate_up_weight'], d['down_weight'], 
            d['eps']
        )
        check_tensor(cpp_out, "C++ output")
    except Exception as e:
        print(f"❌ C++ error: {e}")
        return False
    
    # 对比
    max_err = (py_out - cpp_out).abs().max().item()
    mean_err = (py_out - cpp_out).abs().mean().item()
    
    print(f"\nMax Error:  {max_err:.2e}")
    print(f"Mean Error: {mean_err:.2e}")
    
    # 相对误差（更合理）
    rel_err = ((py_out - cpp_out).abs() / (py_out.abs() + 1e-8)).max().item()
    print(f"Max Relative Error: {rel_err:.2e}")
    
    if max_err < 1e-2 or rel_err < 1e-2:  # fp16 容忍度放宽
        print("✅ PASS")
        return True
    else:
        print("❌ FAIL")
        print(f"Python: {py_out[0,0,:5]}")
        print(f"C++:    {cpp_out[0,0,:5]}")
        return False

def test_perf():
    """性能测试"""
    print("\n⚡ 性能测试")
    print("-" * 50)
    d = make_data()
    N = 1000
    
    # Warmup
    print("Warming up...")
    for _ in range(100):
        MLP.forward(**d)
        # 🔥 修复：补上 eps 参数
        cpp_mlp.forward(
            d['hidden'], d['attn_out'],
            d['attn_proj_weight'], d['norm_weight'],
            d['gate_up_weight'], d['down_weight'],
            d['eps']  # 漏了这个！
        )
    torch.cuda.synchronize()
    
    # Python 测试
    t0 = time.time()
    for _ in range(N):
        MLP.forward(**d)
    torch.cuda.synchronize()
    py_t = (time.time() - t0) / N * 1000
    
    # C++ 测试
    t0 = time.time()
    for _ in range(N):
        # 🔥 修复：补上 eps 参数
        cpp_mlp.forward(
            d['hidden'], d['attn_out'],
            d['attn_proj_weight'], d['norm_weight'],
            d['gate_up_weight'], d['down_weight'],
            d['eps']  # 漏了这个！
        )
    torch.cuda.synchronize()
    cpp_t = (time.time() - t0) / N * 1000
    
    print(f"Python (std_mlp): {py_t:.3f} ms")
    print(f"C++ (cpp_mlp):    {cpp_t:.3f} ms")
    print(f"Speedup:          {py_t/cpp_t:.2f}x")
    
    # 32 层估算
    print("\n📊 32 层整体估算:")
    base = 10
    py_total = base + 32 * py_t
    cpp_total = base + 32 * cpp_t
    print(f"  Python: {py_total:.1f}ms/token ({1000/py_total:.1f} tokens/s)")
    print(f"  C++:    {cpp_total:.1f}ms/token ({1000/cpp_total:.1f} tokens/s)")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ Need CUDA")
        exit(1)
    
    if test_correct():
        test_perf()
    else:
        print("\n功能测试失败")