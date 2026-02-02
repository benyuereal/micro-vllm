import torch
import time
import statistics
from typing import Dict, List, Tuple, Optional
import json

class LayerNormBenchmark:
    """
    LayerNorm性能基准测试类
    专注于评估前向传播性能，支持不同形状、数据类型和迭代次数的测试
    """
    
    def __init__(self, 
                 device: str = 'cuda',
                 warmup_iters: int = 10,
                 measurement_iters: int = 100,
                 precision: str = 'float32'):
        """
        初始化性能测试配置
        
        Args:
            device: 测试设备 ('cuda' 或 'cpu')
            warmup_iters: 预热迭代次数，排除冷启动影响
            measurement_iters: 测量迭代次数
            precision: 测试精度 ('float32', 'float16', 'bfloat16')
        """
        self.device = device
        self.warmup_iters = warmup_iters
        self.measurement_iters = measurement_iters
        self.precision = precision
        
        # 精度映射到torch数据类型
        self.dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16
        }
        self.dtype = self.dtype_map[precision]
        
        # 存储测试结果
        self.results = []
    
    def _create_test_tensors(self, shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        创建测试用的输入张量和参数
        
        Args:
            shape: (M, N) 形状，M=样本数，N=特征维度
            
        Returns:
            (x, weight, bias) 元组
        """
        M, N = shape
        x = torch.randn(M, N, device=self.device, dtype=self.dtype)
        weight = torch.randn(N, device=self.device, dtype=self.dtype)
        bias = torch.randn(N, device=self.device, dtype=self.dtype)
        return x, weight, bias
    
    def _ensure_contiguous(self, *tensors):
        """确保所有张量在内存中是连续的"""
        return [t.contiguous() if t.is_contiguous() else t.contiguous() for t in tensors]
    
    def benchmark_triton(self, 
                        triton_layer_norm_func,
                        shape: Tuple[int, int],
                        eps: float = 1e-5) -> Dict:
        """
        测试Triton实现的性能
        
        Args:
            triton_layer_norm_func: 你的Triton LayerNorm函数
            shape: 测试形状 (M, N)
            eps: LayerNorm的epsilon参数
            
        Returns:
            包含性能指标的字典
        """
        x, weight, bias = self._create_test_tensors(shape)
        x, weight, bias = self._ensure_contiguous(x, weight, bias)
        
        # 预热阶段
        for _ in range(self.warmup_iters):
            _ = triton_layer_norm_func(x, weight, bias, eps)
        
        # 同步GPU确保准确计时
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # 性能测量阶段
        start_time = time.perf_counter()
        for _ in range(self.measurement_iters):
            _ = triton_layer_norm_func(x, weight, bias, eps)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # 计算性能指标
        total_time_ms = (end_time - start_time) * 1000  # 转换为毫秒
        avg_time_ms = total_time_ms / self.measurement_iters
        M, N = shape
        total_elements = M * N * self.measurement_iters
        
        # 计算吞吐量 (元素/秒 和 样本/秒)
        elements_per_sec = total_elements / (total_time_ms / 1000)
        samples_per_sec = M * self.measurement_iters / (total_time_ms / 1000)
        
        return {
            'impl': 'triton',
            'shape': shape,
            'dtype': self.precision,
            'avg_time_ms': avg_time_ms,
            'total_time_ms': total_time_ms,
            'elements_per_sec': elements_per_sec,
            'samples_per_sec': samples_per_sec,
            'throughput_gbps': elements_per_sec * 4 / 1e9 if self.precision == 'float32' else elements_per_sec * 2 / 1e9
        }
    
    def benchmark_pytorch(self, 
                         shape: Tuple[int, int],
                         eps: float = 1e-5) -> Dict:
        """
        测试PyTorch原生实现的性能
        
        Args:
            shape: 测试形状 (M, N)
            eps: LayerNorm的epsilon参数
            
        Returns:
            包含性能指标的字典
        """
        x, weight, bias = self._create_test_tensors(shape)
        x, weight, bias = self._ensure_contiguous(x, weight, bias)
        
        # 预热阶段
        for _ in range(self.warmup_iters):
            _ = torch.nn.functional.layer_norm(x, (shape[1],), weight, bias, eps)
        
        # 同步GPU确保准确计时
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # 性能测量阶段
        start_time = time.perf_counter()
        for _ in range(self.measurement_iters):
            _ = torch.nn.functional.layer_norm(x, (shape[1],), weight, bias, eps)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # 计算性能指标
        total_time_ms = (end_time - start_time) * 1000
        avg_time_ms = total_time_ms / self.measurement_iters
        M, N = shape
        total_elements = M * N * self.measurement_iters
        
        elements_per_sec = total_elements / (total_time_ms / 1000)
        samples_per_sec = M * self.measurement_iters / (total_time_ms / 1000)
        
        return {
            'impl': 'pytorch',
            'shape': shape,
            'dtype': self.precision,
            'avg_time_ms': avg_time_ms,
            'total_time_ms': total_time_ms,
            'elements_per_sec': elements_per_sec,
            'samples_per_sec': samples_per_sec,
            'throughput_gbps': elements_per_sec * 4 / 1e9 if self.precision == 'float32' else elements_per_sec * 2 / 1e9
        }
    
    def run_comparison(self, 
                      triton_layer_norm_func,
                      test_shapes: List[Tuple[int, int]],
                      eps: float = 1e-5) -> List[Dict]:
        """
        运行完整的性能对比测试
        
        Args:
            triton_layer_norm_func: 你的Triton LayerNorm函数
            test_shapes: 要测试的形状列表 [(M1, N1), (M2, N2), ...]
            eps: LayerNorm的epsilon参数
            
        Returns:
            所有测试结果的列表
        """
        print(f"🚀 开始LayerNorm性能测试 (设备: {self.device}, 精度: {self.precision})")
        print(f"预热迭代: {self.warmup_iters}, 测量迭代: {self.measurement_iters}")
        print("=" * 80)
        
        self.results = []
        
        for shape in test_shapes:
            M, N = shape
            print(f"\n📊 测试形状: [{M}, {N}] (总计 {M*N:,} 元素)")
            print("-" * 60)
            
            # 测试PyTorch原生实现
            print("测试 PyTorch 原生实现...")
            torch_result = self.benchmark_pytorch(shape, eps)
            self.results.append(torch_result)
            self._print_result(torch_result)
            
            # 测试Triton实现
            print("测试 Triton 自定义实现...")
            triton_result = self.benchmark_triton(triton_layer_norm_func, shape, eps)
            self.results.append(triton_result)
            self._print_result(triton_result)
            
            # 计算加速比
            speedup = torch_result['avg_time_ms'] / triton_result['avg_time_ms']
            throughput_ratio = triton_result['elements_per_sec'] / torch_result['elements_per_sec']
            
            print(f"\n📈 性能对比:")
            print(f"  加速比 (时间): {speedup:.2f}x")
            print(f"  吞吐量提升: {throughput_ratio:.2f}x")
            print(f"  Triton平均耗时: {triton_result['avg_time_ms']:.3f} ms")
            print(f"  PyTorch平均耗时: {torch_result['avg_time_ms']:.3f} ms")
            
            if speedup > 1.0:
                print(f"  ✅ Triton更快!")
            else:
                print(f"  ⚠️ PyTorch更快或持平")
        
        print("\n" + "=" * 80)
        print("🎯 性能测试完成!")
        return self.results
    
    def _print_result(self, result: Dict):
        """打印单个测试结果"""
        impl = result['impl'].upper()
        print(f"  [{impl}] 平均耗时: {result['avg_time_ms']:.3f} ms")
        print(f"       吞吐量: {result['elements_per_sec']/1e9:.2f} G元素/秒")
        print(f"       带宽: {result['throughput_gbps']:.2f} GB/s")
    
    def export_results(self, filename: str = 'layernorm_benchmark_results.json'):
        """将测试结果导出为JSON文件"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"📁 结果已导出到: {filename}")
    
    def print_summary(self):
        """打印测试结果摘要"""
        print("\n" + "=" * 80)
        print("📋 性能测试摘要")
        print("=" * 80)
        
        # 按形状分组结果
        shape_results = {}
        for result in self.results:
            shape_str = str(result['shape'])
            if shape_str not in shape_results:
                shape_results[shape_str] = []
            shape_results[shape_str].append(result)
        
        for shape_str, results in shape_results.items():
            print(f"\n形状: {shape_str}")
            print("-" * 40)
            
            # 找到Triton和PyTorch的结果
            triton_result = next(r for r in results if r['impl'] == 'triton')
            torch_result = next(r for r in results if r['impl'] == 'pytorch')
            
            speedup = torch_result['avg_time_ms'] / triton_result['avg_time_ms']
            throughput_ratio = triton_result['elements_per_sec'] / torch_result['elements_per_sec']
            
            print(f"  PyTorch: {torch_result['avg_time_ms']:.3f} ms | {torch_result['elements_per_sec']/1e9:.2f} G元素/秒")
            print(f"  Triton:  {triton_result['avg_time_ms']:.3f} ms | {triton_result['elements_per_sec']/1e9:.2f} G元素/秒")
            print(f"  加速比: {speedup:.2f}x | 吞吐量提升: {throughput_ratio:.2f}x")


# 使用示例
if __name__ == "__main__":
    # 导入你已经写好的LayerNorm实现
    # 假设你的实现在一个叫layernorm.py的文件中，有一个layer_norm函数
    try:
        from layernorm import layer_norm as triton_layer_norm
        print("✅ 成功导入Triton LayerNorm实现")
    except ImportError:
        print("⚠️ 无法导入Triton LayerNorm实现，使用一个模拟函数进行演示")
        # 创建一个模拟函数用于演示
        def triton_layer_norm(x, weight, bias, eps=1e-5):
            return torch.nn.functional.layer_norm(x, (x.shape[1],), weight, bias, eps)
    
    # 配置测试参数
    benchmark = LayerNormBenchmark(
        device='cuda',
        warmup_iters=20,
        measurement_iters=100,
        precision='float32'  # 可以改为 'float16' 测试半精度
    )
    
    # 定义要测试的各种形状
    # 格式: (M, N) 其中 M = batch_size * seq_len, N = hidden_size
    test_shapes = [
        (1, 768),       # 小批量，常见hidden_size
        (32, 768),      # 中等批量
        (256, 768),     # 大批量
        (1, 1024),      # 小批量，大hidden_size
        (32, 1024),     # 中等批量，大hidden_size
        (256, 1024),    # 大批量，大hidden_size
        (1024, 4096),    # 类似大模型场景
        (2048, 8192),    # 类似大模型场景

    ]
    
    # 运行性能对比测试
    results = benchmark.run_comparison(
        triton_layer_norm_func=triton_layer_norm,
        test_shapes=test_shapes,
        eps=1e-5
    )
    
    # 打印摘要
    benchmark.print_summary()
    
    # 导出结果
    benchmark.export_results()
    
    # 额外：测试不同精度
    print("\n" + "=" * 80)
    print("🧪 测试不同精度")
    print("=" * 80)
    
    for precision in ['float16', 'float32']:
        print(f"\n精度: {precision}")
        benchmark_fp = LayerNormBenchmark(
            device='cuda',
            warmup_iters=10,
            measurement_iters=50,
            precision=precision
        )
        
        # 只测试一个代表性形状
        test_shape = (32, 1024)
        torch_result = benchmark_fp.benchmark_pytorch(test_shape)
        triton_result = benchmark_fp.benchmark_triton(triton_layer_norm, test_shape)
        
        speedup = torch_result['avg_time_ms'] / triton_result['avg_time_ms']
        print(f"  加速比: {speedup:.2f}x")