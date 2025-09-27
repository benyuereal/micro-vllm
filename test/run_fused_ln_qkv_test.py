#!/usr/bin/env python3
"""
融合LN+QKV GPTQ测试运行脚本
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from test_config_fused_ln_qkv import TEST_CONFIG, TEST_CASES, ERROR_CODES, SUCCESS_CODES

def setup_logging():
    """设置日志"""
    config = TEST_CONFIG["logging"]
    
    # 创建日志格式
    formatter = logging.Formatter(config["format"])
    
    # 创建控制台处理器
    if config["console"]:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(config["file"])
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, config["level"]))

def run_quick_test():
    """运行快速测试"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 运行快速测试...")
    
    try:
        # 导入快速测试
        from quick_fused_ln_qkv_test import quick_test
        
        success = quick_test()
        if success:
            logger.info("✅ 快速测试通过")
            return SUCCESS_CODES["ALL_TESTS_PASSED"]
        else:
            logger.error("❌ 快速测试失败")
            return ERROR_CODES["BASIC_FUNCTIONALITY_FAILED"]
            
    except Exception as e:
        logger.error(f"❌ 快速测试异常: {e}")
        return ERROR_CODES["UNKNOWN_ERROR"]

def run_full_test():
    """运行完整测试"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 运行完整测试...")
    
    try:
        # 导入完整测试
        from run_all_fused_ln_qkv_test import FusedLNQKVTestRunner
        
        # 创建测试运行器
        test_runner = FusedLNQKVTestRunner()
        
        # 运行所有测试
        results = test_runner.run_all_tests()
        
        # 生成报告
        test_runner.generate_report(results)
        
        # 检查结果
        if all(results.values()):
            logger.info("✅ 所有测试通过")
            return SUCCESS_CODES["ALL_TESTS_PASSED"]
        else:
            failed_tests = [name for name, passed in results.items() if not passed]
            logger.error(f"❌ 测试失败: {', '.join(failed_tests)}")
            return ERROR_CODES["BASIC_FUNCTIONALITY_FAILED"]
            
    except Exception as e:
        logger.error(f"❌ 完整测试异常: {e}")
        return ERROR_CODES["UNKNOWN_ERROR"]

def run_specific_test(test_name: str):
    """运行特定测试"""
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 运行特定测试: {test_name}")
    
    try:
        # 导入完整测试
        from run_all_fused_ln_qkv_test import FusedLNQKVTestRunner
        
        # 创建测试运行器
        test_runner = FusedLNQKVTestRunner()
        
        # 检查环境
        if not test_runner.check_environment():
            logger.error("❌ 环境检查失败")
            return ERROR_CODES["CUDA_NOT_AVAILABLE"]
        
        # 运行特定测试
        test_methods = {
            "basic": test_runner.test_basic_functionality,
            "performance": test_runner.test_performance,
            "batch": test_runner.test_batch_processing,
            "correctness": test_runner.test_correctness,
            "integration": test_runner.test_integration,
        }
        
        if test_name not in test_methods:
            logger.error(f"❌ 未知测试: {test_name}")
            return ERROR_CODES["UNKNOWN_ERROR"]
        
        success = test_methods[test_name]()
        if success:
            logger.info(f"✅ {test_name} 测试通过")
            return SUCCESS_CODES["ALL_TESTS_PASSED"]
        else:
            logger.error(f"❌ {test_name} 测试失败")
            return ERROR_CODES["BASIC_FUNCTIONALITY_FAILED"]
            
    except Exception as e:
        logger.error(f"❌ {test_name} 测试异常: {e}")
        return ERROR_CODES["UNKNOWN_ERROR"]

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="融合LN+QKV GPTQ测试运行器")
    parser.add_argument(
        "--mode", 
        choices=["quick", "full", "specific"],
        default="quick",
        help="测试模式"
    )
    parser.add_argument(
        "--test",
        choices=["basic", "performance", "batch", "correctness", "integration"],
        help="特定测试名称（仅在specific模式下使用）"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    logger = logging.getLogger(__name__)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("🚀 开始融合LN+QKV GPTQ测试")
    
    # 根据模式运行测试
    if args.mode == "quick":
        return run_quick_test()
    elif args.mode == "full":
        return run_full_test()
    elif args.mode == "specific":
        if not args.test:
            logger.error("❌ 特定测试模式需要指定测试名称")
            return ERROR_CODES["UNKNOWN_ERROR"]
        return run_specific_test(args.test)
    else:
        logger.error(f"❌ 未知测试模式: {args.mode}")
        return ERROR_CODES["UNKNOWN_ERROR"]

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
