import torch
from reprobe import Interceptor
import torch

def test_interceptor_explicit_end_layer():
    target_end_layer = 200
    
    class FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def register_forward_hook(self, fn): pass
    
    class FakeModel:
        class model:
            layers = torch.nn.ModuleList([FakeLayer() for _ in range(target_end_layer)])
    interceptor = Interceptor(FakeModel(), end_layer=target_end_layer)
    interceptor.attach()
    
    assert interceptor.end_layer == target_end_layer