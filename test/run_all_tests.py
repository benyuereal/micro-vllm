#!/usr/bin/env python3
"""
主测试脚本 - 运行所有融合内核测试
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_test(test_dir, test_script):
    """运行单个测试"""
    test_path = Path(__file__).parent / test_dir / test_script
    
    if not test_path.exists():
        logger.error(f"❌ 测试脚本不存在: {test_path}")
        return False
    
    logger.info(f"🚀 运行测试: {test_dir}/{test_script}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_path)],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        if result.returncode == 0:
            logger.info(f"✅ 测试通过: {test_dir}/{test_script}")
            return True
        else:
            logger.error(f"❌ 测试失败: {test_dir}/{test_script}")
            logger.error(f"错误输出: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ 测试超时: {test_dir}/{test_script}")
        return False
    except Exception as e:
        logger.error(f"❌ 测试异常: {test_dir}/{test_script} - {e}")
        return False

def main():
    """主函数"""
    logger.info("🚀 开始运行所有融合内核测试...")
    
    # 测试列表
    tests = [
        ("vllm_fusion", "fusion.py"),
        ("ln_qkv_fusion", "fusion.py"),
    ]
    
    # 运行测试
    results = []
    for test_dir, test_script in tests:
        success = run_test(test_dir, test_script)
        results.append((test_dir, test_script, success))
    
    # 输出结果
    logger.info("\n📊 测试结果汇总:")
    logger.info("=" * 50)
    
    all_passed = True
    for test_dir, test_script, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        logger.info(f"{test_dir}/{test_script}: {status}")
        if not success:
            all_passed = False
    
    logger.info("=" * 50)
    
    if all_passed:
        logger.info("🎉 所有测试通过!")
    else:
        logger.error("❌ 部分测试失败")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)