#!/usr/bin/env python3
"""
GPU驱动测试程序
用于验证NVIDIA驱动是否正确安装和配置
"""

import sys
import subprocess
import time

def check_nvidia_smi():
    """检查nvidia-smi命令是否可用"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ nvidia-smi 命令可用")
            print("GPU信息:")
            print(result.stdout)
            return True
        else:
            print("✗ nvidia-smi 命令失败:")
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("✗ nvidia-smi 命令未找到，请检查NVIDIA驱动安装")
        return False

def test_cuda_basic():
    """测试基本的CUDA功能"""
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA可用，设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
            
            # 简单的GPU计算测试
            device = torch.cuda.current_device()
            print(f"\n✓ 当前使用GPU: {device}")
            
            # 创建测试张量
            x = torch.rand(1000, 1000).cuda()
            y = torch.rand(1000, 1000).cuda()
            
            start_time = time.time()
            z = torch.matmul(x, y)
            print(z)
            torch.cuda.synchronize()  # 等待GPU完成计算
            end_time = time.time()
            
            print(f"✓ GPU矩阵乘法测试完成")
            print(f"  计算时间: {(end_time - start_time)*1000:.2f} ms")
            print(f"  结果形状: {z.shape}")
            print(f"  结果校验: 平均值={z.mean().item():.6f}")
            
            return True
        else:
            print("✗ CUDA不可用")
            return False
            
    except ImportError:
        print("✗ PyTorch未安装，无法测试CUDA功能")
        print("  可以运行: pip install torch torchvision")
        return False
    except Exception as e:
        print(f"✗ CUDA测试失败: {e}")
        return False

def test_tensorflow_gpu():
    """测试TensorFlow GPU支持"""
    try:
        import tensorflow as tf
        print(f"\n✓ TensorFlow版本: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ TensorFlow检测到 {len(gpus)} 个GPU:")
            for gpu in gpus:
                print(f"  - {gpu}")
                
            # 简单的GPU计算测试
            with tf.device('/GPU:0'):
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                c = tf.matmul(a, b)
                
            print(f"✓ TensorFlow GPU计算测试完成")
            print(f"  结果形状: {c.shape}")
            return True
        else:
            print("✗ TensorFlow未检测到GPU")
            return False
            
    except ImportError:
        print("✗ TensorFlow未安装")
        return False
    except Exception as e:
        print(f"✗ TensorFlow GPU测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("NVIDIA GPU驱动测试程序")
    print("=" * 60)
    
    # 检查nvidia-smi
    nvidia_ok = check_nvidia_smi()
    
    print("\n" + "=" * 40)
    print("CUDA功能测试 (PyTorch)")
    print("=" * 40)
    cuda_ok = test_cuda_basic()
    
    print("\n" + "=" * 40)
    print("TensorFlow GPU测试")
    print("=" * 40)
    tf_ok = test_tensorflow_gpu()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print("=" * 60)
    print(f"nvidia-smi: {'✓ 正常' if nvidia_ok else '✗ 异常'}")
    print(f"CUDA/PyTorch: {'✓ 正常' if cuda_ok else '✗ 异常'}")
    print(f"TensorFlow GPU: {'✓ 正常' if tf_ok else '✗ 异常'}")
    
    if nvidia_ok and cuda_ok:
        print("\n🎉 GPU驱动看起来工作正常！")
    elif nvidia_ok:
        print("\n⚠️  nvidia-smi正常，但深度学习框架可能有配置问题")
    else:
        print("\n❌ GPU驱动可能存在问题，建议重新检查安装")

if __name__ == "__main__":
    main()