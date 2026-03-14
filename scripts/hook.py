import logging

logger = logging.getLogger(__name__)
class Hook():
    def __init__(self, model):
        self.handles = []
        self.model = model
    
    def _get_layers_to_hook(self):
        raise NotImplementedError
    
    def attach(self):
        for layer_idx, data in self._get_layers_to_hook():
            handle = self.model.model.layers[layer_idx].register_forward_hook(
                self._get_hook(layer_idx, data)
            )
            self.handles.append(handle)

        return self
            
    def detach(self):
        for handle in self.handles:
            handle.remove()