#!/usr/bin/env python3
"""
融合LN+QKV GPTQ测试配置
"""

# 测试配置
TEST_CONFIG = {
    # 基础配置
    "hidden_dim": 4096,
    "num_heads": 32,
    "kv_num_heads": 32,
    "head_size": 128,
    "groupsize": 128,
    "eps": 1e-6,
    
    # 性能目标
    "performance_targets": {
        "fused_ln_qkv": 0.25,  # ms
        "gptq_qkv": 0.10,      # ms
        "layernorm": 0.05,     # ms
        "speedup_vs_separate": 2.0,  # x
        "speedup_vs_pytorch": 5.0,   # x
    },
    
    # 测试数据配置
    "test_configs": [
        {"batch_size": 1, "seq_len": 1, "name": "单token"},
        {"batch_size": 1, "seq_len": 16, "name": "短序列"},
        {"batch_size": 1, "seq_len": 64, "name": "中序列"},
        {"batch_size": 1, "seq_len": 256, "name": "长序列"},
        {"batch_size": 4, "seq_len": 16, "name": "批处理"},
    ],
    
    # 性能测试配置
    "performance_test": {
        "num_runs": 100,
        "warmup_runs": 10,
        "batch_test_runs": 50,
    },
    
    # 正确性测试配置
    "correctness_test": {
        "tolerance": 1e-3,
        "check_nan": True,
        "check_inf": True,
        "check_shape": True,
    },
    
    # 批处理测试配置
    "batch_test": {
        "batch_sizes": [1, 2, 4, 8],
        "seq_lens": [1, 16, 32, 64],
        "hidden_dims": [4096],
    },
    
    # 集成测试配置
    "integration_test": {
        "test_layers": [1, 2, 4, 8],
        "test_heads": [32],
        "test_head_sizes": [128],
    },
    
    # 编译配置
    "compile_config": {
        "cuda_archs": [
            "compute_70,code=sm_70",
            "compute_75,code=sm_75", 
            "compute_80,code=sm_80",
            "compute_86,code=sm_86"
        ],
        "nvcc_flags": "-O3 -use_fast_math -lineinfo --ptxas-options=-v",
        "include_paths": [],
        "library_paths": [],
    },
    
    # 日志配置
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(levelname)s - %(message)s",
        "file": "fused_ln_qkv_test.log",
        "console": True,
    },
    
    # 报告配置
    "report": {
        "format": "markdown",
        "file": "fused_ln_qkv_test_report.md",
        "include_performance": True,
        "include_correctness": True,
        "include_batch": True,
        "include_integration": True,
    },
}

# 测试用例
TEST_CASES = {
    "basic_functionality": {
        "name": "基础功能测试",
        "description": "测试融合算子的基本功能",
        "required": True,
        "timeout": 30,  # seconds
    },
    "performance": {
        "name": "性能测试",
        "description": "测试融合算子的性能",
        "required": True,
        "timeout": 60,
    },
    "batch_processing": {
        "name": "批处理测试",
        "description": "测试不同批处理配置",
        "required": True,
        "timeout": 45,
    },
    "correctness": {
        "name": "正确性测试",
        "description": "测试输出正确性",
        "required": True,
        "timeout": 30,
    },
    "integration": {
        "name": "集成测试",
        "description": "测试集成功能",
        "required": True,
        "timeout": 30,
    },
}

# 错误代码
ERROR_CODES = {
    "CUDA_NOT_AVAILABLE": 1,
    "KERNEL_NOT_FOUND": 2,
    "COMPILE_FAILED": 3,
    "BASIC_FUNCTIONALITY_FAILED": 4,
    "PERFORMANCE_FAILED": 5,
    "BATCH_PROCESSING_FAILED": 6,
    "CORRECTNESS_FAILED": 7,
    "INTEGRATION_FAILED": 8,
    "UNKNOWN_ERROR": 99,
}

# 成功代码
SUCCESS_CODES = {
    "ALL_TESTS_PASSED": 0,
    "PARTIAL_SUCCESS": 10,
}

# 测试状态
TEST_STATUS = {
    "PENDING": "pending",
    "RUNNING": "running", 
    "PASSED": "passed",
    "FAILED": "failed",
    "SKIPPED": "skipped",
    "TIMEOUT": "timeout",
}

# 性能指标
PERFORMANCE_METRICS = {
    "latency": "ms",
    "throughput": "tokens/s",
    "memory_usage": "MB",
    "gpu_utilization": "%",
    "speedup": "x",
}

# 测试环境要求
ENVIRONMENT_REQUIREMENTS = {
    "cuda": {
        "min_version": "11.0",
        "required": True,
    },
    "pytorch": {
        "min_version": "1.12.0",
        "required": True,
    },
    "python": {
        "min_version": "3.8",
        "required": True,
    },
    "gpu_memory": {
        "min_mb": 4096,
        "required": True,
    },
}

# 测试数据生成器
class TestDataGenerator:
    """测试数据生成器"""
    
    @staticmethod
    def create_input_data(batch_size: int, seq_len: int, hidden_dim: int) -> torch.Tensor:
        """创建输入数据"""
        return torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16)
    
    @staticmethod
    def create_gptq_params(input_dim: int, output_dim: int, groupsize: int = 128):
        """创建GPTQ参数"""
        qweight = torch.randint(0, 255, (input_dim // 8, output_dim), dtype=torch.uint32)
        qzeros = torch.randint(0, 255, (output_dim // groupsize, groupsize // 8), dtype=torch.uint32)
        scales = torch.randn(output_dim // groupsize, output_dim, dtype=torch.float16)
        return qweight, qzeros, scales
    
    @staticmethod
    def create_layernorm_params(hidden_dim: int):
        """创建LayerNorm参数"""
        weight = torch.randn(hidden_dim, dtype=torch.float16)
        bias = torch.randn(hidden_dim, dtype=torch.float16)
        return weight, bias

# 性能分析器
class PerformanceAnalyzer:
    """性能分析器"""
    
    @staticmethod
    def analyze_times(times: List[float]) -> Dict[str, float]:
        """分析时间数据"""
        return {
            "mean": np.mean(times),
            "median": np.median(times),
            "min": np.min(times),
            "max": np.max(times),
            "std": np.std(times),
            "p95": np.percentile(times, 95),
            "p99": np.percentile(times, 99),
        }
    
    @staticmethod
    def calculate_speedup(baseline_times: List[float], optimized_times: List[float]) -> float:
        """计算加速比"""
        baseline_mean = np.mean(baseline_times)
        optimized_mean = np.mean(optimized_times)
        return baseline_mean / optimized_mean if optimized_mean > 0 else 0.0

# 测试结果验证器
class TestResultValidator:
    """测试结果验证器"""
    
    @staticmethod
    def validate_output_shape(output: torch.Tensor, expected_shape: Tuple[int, ...]) -> bool:
        """验证输出形状"""
        return output.shape == expected_shape
    
    @staticmethod
    def validate_output_values(output: torch.Tensor, check_nan: bool = True, check_inf: bool = True) -> bool:
        """验证输出值"""
        if check_nan and torch.any(torch.isnan(output)):
            return False
        if check_inf and torch.any(torch.isinf(output)):
            return False
        return True
    
    @staticmethod
    def validate_performance(actual_time: float, target_time: float) -> bool:
        """验证性能"""
        return actual_time <= target_time

# 导出配置
__all__ = [
    "TEST_CONFIG",
    "TEST_CASES", 
    "ERROR_CODES",
    "SUCCESS_CODES",
    "TEST_STATUS",
    "PERFORMANCE_METRICS",
    "ENVIRONMENT_REQUIREMENTS",
    "TestDataGenerator",
    "PerformanceAnalyzer",
    "TestResultValidator",
]
