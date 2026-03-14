import logging
from typing import Callable, Literal, TypedDict
import torch
from datasets import load_dataset

from scripts.hook import Hook

logger = logging.getLogger(__name__)

class ActivationDataset(TypedDict):
    acts: torch.Tensor
    labels: torch.Tensor

class Interceptor(Hook):
    def __init__(self, model, start_layer: int = 0, end_layer: int = None, training_mode: Literal["prefill", "token"] = "token"):
        super().__init__(model)
        
        self.training_mode = training_mode
        self.activations = []
        self._capture_next = False  
        self._acts_buffer = {} # Utilisation d'un dict pour garantir l'ordre des layers
        
        self.start_layer = start_layer
        self.end_layer = end_layer if end_layer is not None else len(model.model.layers)
        
    def _get_layers_to_hook(self):
        # On passe None en data car on n'en a pas besoin pour juste intercepter
        return [(i, None) for i in range(self.start_layer, self.end_layer)]

    def _get_hook(self, layer_idx, data): # Ajout de 'data' pour respecter le contrat
        def _hook_fn(module, input, output):
            if not self._capture_next:
                return 
            
            # Détection du mode (Prompt vs Token)
            hidden_states = output[0] if isinstance(output, tuple) else output
            is_prefill = hidden_states.shape[1] > 1 #if prefill is true, we're in the prefill moment

            if self.training_mode == "token":
                if is_prefill:
                    return # ignore prefill moment
            else:
                if not is_prefill: #get only prefill
                    self._flush()
                    return   
                
            # <Capture last token
            self._acts_buffer[layer_idx] = hidden_states[:, -1, :].detach().cpu()
            
            # Si on a atteint la dernière couche, on range ça dans les activations finales
            if len(self._acts_buffer) == (self.end_layer - self.start_layer):
                self._flush()
                
        return _hook_fn
    
    def allow_one_capture(self):
        self._capture_next = True
        return self  
    
    def _flush(self, block_capture=True):
        if self._acts_buffer:
            # On trie par index de couche pour garantir l'ordre dans le stack
            sorted_layers = sorted(self._acts_buffer.keys())
            acts = [self._acts_buffer[l] for l in sorted_layers]
            
            stacked = torch.stack(acts, dim=0)  # [num_layers, batch, hidden_dim]
            stacked = stacked.permute(1, 0, 2)   # [batch, num_layers, hidden_dim]
            
            for i in range(stacked.shape[0]):
                self.activations.append(stacked[i].unsqueeze(0))
            
            self._acts_buffer = {}
            if block_capture:
                self._capture_next = False
    

class Classifier():
    """
    Contrat : classify(text) -> torch.Tensor de shape [batch], valeurs dans [0, 1]
    normalize_output doit respecter ce contrat.
    """
    def __init__(self, model, device, normalize_output: None | Callable = None, tokenizer = None):
        self.model = model.to(device)
        self.normalize_output = normalize_output
        self.tokenizer = tokenizer
        self.device = device
        
    def classify(self, text):
        if self.tokenizer:
            encoded = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        else:
            encoded = self.model.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        out = self.model(**encoded)
        results = self.normalize_output(out) if (self.normalize_output) else out
        
        return results
    
class DatasetConfig(TypedDict):
    type: Literal["huggingface", "json"]
    name: str
    fn_data: Callable
    params: dict
    
class DataManager():
    def __init__(self, datasets: list[DatasetConfig]):
        self.datasets = datasets
        
    def load(self):
        datas = []
        for dataset in self.datasets:
            cfg = dataset.copy()
            fn = cfg.pop("fn_data")
            if cfg["type"] == "huggingface":
                ds = load_dataset(cfg.pop("name"), **cfg["params"])
                datas.append(fn(ds))