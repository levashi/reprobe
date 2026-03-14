
import torch

from reprobe import Steerer


def test_steerer_projected():
    hidden_dim = 8
    direction = torch.zeros(hidden_dim)
    direction[0] = 1.0  # pointe uniquement sur la dimension 0

    orthogonal = torch.zeros(hidden_dim)
    orthogonal[1] = 1.0  # pointe uniquement sur la dimension 1
    
    hidden = Steerer._apply_projection(orthogonal, direction, alpha=1)
    
    assert torch.allclose(hidden, orthogonal)