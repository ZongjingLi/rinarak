#from karanir import *
import torch
from torch import nn

class BoxRegistry(nn.Module):
    _init_methods = {"uniform": torch.nn.init.uniform_}

    def __init__(self, dim, entries, center = [0.1, -0.1], offset = [0.1,0.5]):
        super().__init__()
        self.dim = dim

        entries = entries
        self.entries = entries

        init_config = config
        self.boxes = self._init_embedding_(entries, init_config)
        
        self.offset_clamp = config.offset
        self.offset_clamp = config.center
    
    def _init_embedding_(self, entries, config):
        init_method = self._init_methods["uniform"]#config.method
        center = torch.Tensor(entries, self.dim)
        offset = torch.Tensor(entries, self.dim)
        self._init_methods[init_method](center, *config.center)
        self._init_methods[init_method](offset, *config.offset)
        return nn.Embedding(entries, self.dim * 2, _weight=torch.cat([center, offset], dim=1))
    
    def forward(self, x):
        embs = self.boxes(x)
        return embs
        return torch.cat( [torch.tanh(embs[:,:self.dim]) * 0.5,\
                           torch.sigmoid(embs[:,:self.dim:]) * 0.5 ], dim = -1 )


    def clamp_dimensions(self):
        with torch.no_grad():
            self.boxes.weight[:, self.dim:].clamp_(*self.offset_clamp)
            self.boxes.weight[:, :self.dim].clamp_(*self.center_clamp)

    def __getitem__(self, key, item): self.boxes.weight[key] = item

    def __getitem__(self, key):return torch.tanh(self.boxes.weight[key])*0.25

    @property
    def device(self):return self.boxes.weight.device

    @property
    def prototypes(self):return self.boxes.weight.detach()

    def __len__(self):return len(self.boxes.weight)

    @property
    def size(self):return self.dim ** 2


class PlaneRegistry(nn.Module):

    def __init__(self, dim, entries):
        super().__init__()
        self.dim = dim
        self.planes = nn.Embedding(entries, self.dim)
        with torch.no_grad():
            self.planes.weight.abs_()

    def forward(self, x):
        return torch.tanh(self.planes(x) ) * 0.5

    def __setitem__(self, key, item):
        self.planes.weight[key] = item

    def __getitem__(self, key):
        return self.planes.weight[key]

    def clamp_dimensions(self):
        with torch.no_grad():
            self.planes.weight.clamp(0, 1)
        pass

    @property
    def device(self):
        return self.planes.weight.device

    def __len__(self):
        return len(self.planes.weight)

    @property
    def size(self):
        return self.dim


class ConeRegistry(nn.Module):

    def __init__(self, dim, entries):
        super().__init__()
        self.dim = dim
        self.cones = self._init_embedding_(entries)

    def _init_embedding_(self, entries):
        weight = torch.Tensor(entries, self.dim).normal_().abs_()
        return nn.Embedding(entries, self.dim, _weight=weight)

    def forward(self, x):
        return torch.tanh(self.cones(x)) 

    def __setitem__(self, key, item):
        self.cones.weight[key] = item

    def __getitem__(self, key):
        return torch.tanh(self.cones.weight[key]) * 0.5

    def clamp_dimensions(self):
        with torch.no_grad():
            self.cones.weight /= self.cones.weight.norm(dim=-1, keepdim=True)
        pass

    @property
    def device(self):
        return self.cones.weight.device

    def __len__(self):
        return len(self.cones.weight)

    @property
    def size(self):
        return self.dim


registry_map = {"box": BoxRegistry, "cone": ConeRegistry, "plane": PlaneRegistry, }

def build_box_registry(concept_type, dim, entries):return registry_map[concept_type](dim, entries)