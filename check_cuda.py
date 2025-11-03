"""检验 CUDA 环境是否正确安装"""

import sys

def check_cuda():
    print("=" * 50)
    print("CUDA 环境检测")
    print("=" * 50)
    
    # 检查 PyTorch
    try:
        import torch
        print(f"\n✓ PyTorch 版本: {torch.__version__}")
        print(f"✓ CUDA 是否可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA 版本: {torch.version.cuda}")
            print(f"✓ cuDNN 版本: {torch.backends.cudnn.version()}")
            print(f"✓ GPU 数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"✓ GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  - 显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        else:
            print("✗ CUDA 不可用 (使用 CPU 模式)")
            
    except ImportError:
        print("\n✗ PyTorch 未安装")
        return False
    
    # 简单的 CUDA 测试
    if torch.cuda.is_available():
        try:
            print("\n" + "=" * 50)
            print("执行简单的 CUDA 测试...")
            print("=" * 50)
            
            # 创建一个张量并移动到 GPU
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = x + y
            
            print(f"✓ GPU 计算测试成功")
            print(f"  测试张量设备: {z.device}")
            
        except Exception as e:
            print(f"✗ GPU 计算测试失败: {e}")
            return False
    
    print("\n" + "=" * 50)
    print("检测完成!")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = check_cuda()
    sys.exit(0 if success else 1)
