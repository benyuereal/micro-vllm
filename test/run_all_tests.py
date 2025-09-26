#!/usr/bin/env python3
"""
一键测试脚本
测试CUDA融合内核功能和性能
"""

import sys
import os
import subprocess

def run_test(test_name, test_file):
    """运行单个测试"""
    print(f"\n🚀 运行 {test_name}")
    print("=" * 50)
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {test_name} 测试通过!")
            print(result.stdout)
        else:
            print(f"❌ {test_name} 测试失败!")
            print(f"错误输出: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_name} 测试超时!")
        return False
    except Exception as e:
        print(f"❌ {test_name} 测试异常: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("🚀 CUDA融合内核一键测试")
    print("=" * 60)
    
    # 测试列表
    tests = [
        ("导入测试", "test_import.py"),
        ("简单功能测试", "simple_test.py"),
        ("GPTQ功能测试", "test_gptq_functionality.py"),
        ("批处理Layer层测试", "test_batch_layer_shapes.py"),
        ("Layer层集成测试", "test_layer_integration.py"),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_file in tests:
        test_path = os.path.join(os.path.dirname(__file__), test_file)
        
        if not os.path.exists(test_path):
            print(f"⚠️ 测试文件不存在: {test_path}")
            continue
        
        if run_test(test_name, test_path):
            passed_tests += 1
    
    # 测试结果汇总
    print(f"\n📊 测试结果汇总")
    print("=" * 60)
    print(f"通过测试: {passed_tests}/{total_tests}")
    print(f"通过率: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 所有测试都通过!")
    elif passed_tests > 0:
        print("✅ 部分测试通过")
    else:
        print("❌ 所有测试都失败")
    
    print("\n🎉 一键测试完成!")

if __name__ == "__main__":
    main()
