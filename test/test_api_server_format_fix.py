#!/usr/bin/env python3
"""
快速测试API服务器中的GPTQ格式修复
验证qzeros [32, 1536] 和 scales [32, 12288] 格式的处理
"""

import torch
import sys
import os
import logging

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.layer.gptq import GPTQCUDAFusion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api_server_format():
    """测试API服务器中的GPTQ格式"""
    print("🚀 测试API服务器中的GPTQ格式修复")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试。")
        return
    
    # 创建GPTQ融合实例
    gptq_fusion = GPTQCUDAFusion(groupsize=128)
    print("✅ GPTQ CUDA融合实例创建成功")
    
    # 模拟API服务器中的实际数据 - 使用错误日志中的确切格式
    M, K, N = 1, 4096, 12288
    
    # 输入数据
    input_tensor = torch.randn(M, K, dtype=torch.float16, device='cuda')
    
    # 模拟API服务器中的GPTQ格式 - 使用错误日志中的确切维度
    qzeros_api = torch.randint(0, 16, (32, 1536), dtype=torch.uint32, device='cuda')
    scales_api = torch.randn(32, 12288, dtype=torch.float16, device='cuda')  # 注意：这里是12288
    qweight_api = torch.randint(0, 256, (512, 12288), dtype=torch.uint32, device='cuda')
    
    print(f"📊 API服务器格式 (来自错误日志):")
    print(f"  input: {input_tensor.shape}")
    print(f"  qweight: {qweight_api.shape}")
    print(f"  qzeros: {qzeros_api.shape}")
    print(f"  scales: {scales_api.shape}")
    
    try:
        # 测试CUDA内核
        output = gptq_fusion.fused_gptq_gemm_4bit(
            input=input_tensor,
            qweight=qweight_api,
            qzeros=qzeros_api,
            scales=scales_api
        )
        
        print(f"✅ API服务器格式修复测试通过!")
        print(f"📊 输出形状: {output.shape}")
        print(f"📊 期望形状: torch.Size([{M}, {N}])")
        
        assert output.shape == (M, N), f"输出形状错误: {output.shape}"
        
    except Exception as e:
        print(f"❌ API服务器格式修复测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 API服务器格式修复测试完成!")
    return True

if __name__ == "__main__":
    print("🚀 开始API服务器格式修复测试")
    
    success = test_api_server_format()
    
    if success:
        print("🎯 API服务器格式修复成功!")
    else:
        print("⚠️ API服务器格式修复失败")
