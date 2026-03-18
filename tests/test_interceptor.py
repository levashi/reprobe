import torch
from reprobe import Interceptor
import torch

def test_flush_layer_order():
    hidden_dim = 16
    batch = 5
    ids = [0.9678, 1.87]
    mock_model = {"model": {"layers": [i for i in range(200)]}}
    interceptor = Interceptor(mock_model, end_layer=200, training_mode="all")

    # prefill
    interceptor._acts_buffer[15] = torch.full((batch, hidden_dim), ids[0])
    interceptor._acts_buffer[12] = torch.full((batch, hidden_dim), ids[1])
    interceptor._flush("prefill")

    # token
    interceptor.allow_one_capture(batch)
    interceptor._acts_buffer[15] = torch.full((batch, hidden_dim), ids[0])
    interceptor._acts_buffer[12] = torch.full((batch, hidden_dim), ids[1])
    interceptor._flush("token")

    acts = interceptor.finalize()
    
    
    assert torch.allclose(acts["prefill"][0, 0, :], torch.full((hidden_dim,), ids[1]))
    assert torch.allclose(acts["prefill"][0, 1, :], torch.full((hidden_dim,), ids[0]))
    
    assert torch.allclose(acts["token"][0][0, 0, :], torch.full((hidden_dim,), ids[1]))
    assert torch.allclose(acts["token"][0][0, 1, :], torch.full((hidden_dim,), ids[0]))
        
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

def test_align():
    hidden_dim = 16
    num_layers = 2

    class MockClassifier:
        def classify(self, texts):
            # simule un score fixe par texte pour pouvoir asserter
            return torch.tensor([0.9]) if "toxic" in texts[0] else torch.tensor([0.1])

    interceptor = Interceptor(None, end_layer=2, training_mode="token")

    # 3 tokens 
    interceptor.allow_one_capture(1)
    for _ in range(3):
        interceptor._current_token_prompts[0].append(torch.zeros(num_layers, hidden_dim))

    # 2 tokens 
    interceptor.allow_one_capture(1)
    for _ in range(2):
        interceptor._current_token_prompts[0].append(torch.zeros(num_layers, hidden_dim))

    texts = ["toxic text", "safe text"]
    result = interceptor.finalize()
    token_acts, labels = Interceptor.align(result, texts=texts, classifier=MockClassifier())
    
    result = {
        "prefill": result["prefill"],
        "token": token_acts
    }
    assert result["token"].shape == (5, num_layers, hidden_dim)  # 3 + 2 tokens
    assert labels.shape == (5,)
    assert torch.allclose(labels[:3], torch.tensor([0.9, 0.9, 0.9]))
    assert torch.allclose(labels[3:], torch.tensor([0.1, 0.1]))