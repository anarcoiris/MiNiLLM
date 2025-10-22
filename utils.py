import sys

def get_device(prefer_cuda: bool = True):
    """Detecta y retorna device disponible"""
    try:
        import torch
        
        if not prefer_cuda:
            return 'cpu'
        
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ Usando GPU: {gpu_name} ({memory_gb:.1f} GB)")
        else:
            device = 'cpu'
            print("ℹ️  GPU no disponible, usando CPU")
            print("   Esto será MUY lento. Considera usar Google Colab.")
        
        return device
    except ImportError:
        return 'cpu'