#!/usr/bin/env python3
"""
GPU Detection and Capabilities Utility

This utility detects available GPU acceleration for PyTorch training,
including Apple Metal (MPS), NVIDIA CUDA, and fallback to CPU.
Used by DQN and other deep learning agents to optimize training performance.

Usage:
    python -m utils.gpu_detection
"""

import torch
import subprocess
import platform

def check_gpu_support():
    """Check what GPU support is available."""
    print("🖥️  GPU Support Detection for Mac Mini")
    print("=" * 50)
    
    # System info
    print(f"🖥️  System: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {platform.python_version()}")
    print(f"🔥 PyTorch: {torch.__version__}")
    
    print(f"\n💾 Device Detection:")
    
    # Check CPU
    print(f"   ✅ CPU: Available")
    print(f"      Device: {torch.device('cpu')}")
    
    # Check CUDA (NVIDIA GPUs)
    cuda_available = torch.cuda.is_available()
    print(f"   {'✅' if cuda_available else '❌'} CUDA (NVIDIA): {cuda_available}")
    if cuda_available:
        print(f"      Devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"      GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Check MPS (Apple Silicon Metal Performance Shaders)
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    print(f"   {'✅' if mps_available else '❌'} MPS (Apple Metal): {mps_available}")
    if mps_available:
        print(f"      Device: {torch.device('mps')}")
    
    # Determine best device
    if mps_available:
        best_device = "mps"
        device_emoji = "🍎"
        device_desc = "Apple Metal Performance Shaders"
    elif cuda_available:
        best_device = "cuda"
        device_emoji = "🟢"
        device_desc = "NVIDIA CUDA GPU"
    else:
        best_device = "cpu" 
        device_emoji = "💻"
        device_desc = "CPU (no GPU acceleration)"
    
    print(f"\n🏆 Recommended Device: {device_emoji} {best_device}")
    print(f"   Description: {device_desc}")
    
    # Test tensor operations
    print(f"\n🧪 Testing Tensor Operations:")
    device = torch.device(best_device)
    
    try:
        # Create test tensors
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        
        # Time matrix multiplication
        import time
        start = time.time()
        for _ in range(10):
            z = torch.mm(x, y)
        end = time.time()
        
        print(f"   ✅ Matrix multiply test passed on {device}")
        print(f"   ⏱️  Time: {(end-start)*1000:.1f}ms for 10 operations")
        
    except Exception as e:
        print(f"   ❌ Error testing {device}: {e}")
        print(f"   💡 Falling back to CPU")
        best_device = "cpu"
    
    # Mac-specific info
    if platform.system() == "Darwin":
        try:
            # Get Mac model info
            result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Model Name' in line or 'Chip' in line or 'Model Identifier' in line:
                        print(f"   🖥️  {line.strip()}")
        except:
            pass
    
    return best_device

def get_performance_comparison():
    """Compare performance between CPU and MPS if available."""
    print(f"\n⚡ Performance Comparison:")
    
    devices_to_test = ['cpu']
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices_to_test.append('mps')
    if torch.cuda.is_available():
        devices_to_test.append('cuda')
    
    results = {}
    
    for device_name in devices_to_test:
        device = torch.device(device_name)
        try:
            # Neural network forward pass test (similar to DQN)
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 2)
            ).to(device)
            
            # Test batch
            x = torch.randn(32, 4).to(device)
            
            import time
            start = time.time()
            for _ in range(1000):
                output = model(x)
            end = time.time()
            
            time_ms = (end - start) * 1000
            results[device_name] = time_ms
            
            print(f"   {device_name.upper():>4}: {time_ms:.1f}ms (1000 forward passes)")
            
        except Exception as e:
            print(f"   {device_name.upper():>4}: Error - {e}")
    
    # Show speedup
    if len(results) > 1 and 'cpu' in results:
        cpu_time = results['cpu']
        for device, time_ms in results.items():
            if device != 'cpu':
                speedup = cpu_time / time_ms
                print(f"   📈 {device.upper()} is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} than CPU")

if __name__ == "__main__":
    best_device = check_gpu_support()
    get_performance_comparison()
    
    print(f"\n💡 For DQN Training:")
    print(f"   Use device: '{best_device}'")
    print(f"   Expected benefit: {'GPU acceleration for neural networks' if best_device != 'cpu' else 'CPU-only, slower but still works'}")
    
    if best_device == 'mps':
        print(f"   🍎 Your Mac Mini with Apple Silicon supports Metal GPU acceleration!")
        print(f"   🚀 This can speed up training by 2-5x compared to CPU")
    elif best_device == 'cuda':
        print(f"   🟢 NVIDIA GPU detected - excellent for deep learning!")
    else:
        print(f"   💻 CPU-only training - still works but slower for large models")