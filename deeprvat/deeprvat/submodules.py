
import copy
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
from deeprvat.utils import pad_variants

NORMALIZATION = {"spectral_norm":  nn.utils.parametrizations.spectral_norm,
                 "LayerNorm":  nn.LayerNorm}

class ModelAverage(nn.Module):
    def __init__(self, model):
        super(ModelAverage, self).__init__()
        self.model = model
        self.beta = model.hparams.moving_avg['beta']
        self.start_iteration = model.hparams.moving_avg['start_iteration']
        
        self.averaged_model = copy.deepcopy(self.model)
        # needs to be set because the averaged model does not have an averaged model
        # so in stage=val its forward operation would raise an NotImplementedEerror
        self.averaged_model.model_avg = False
        if next(self.model.parameters()).is_cuda: self.averaged_model.cuda()
        self.averaged_model.eval()
        for p in self.averaged_model.parameters(): 
            p.requires_grad = False
        
        self.num_updates = 0

    # dont change averaged model to training
    def train(self,mode: bool = True):
        self.training = mode
        for module in self.model.children():
            module.train(mode)
        return self
    
    def forward(self, *inputs, **kwargs):
        if self.training: return self.model(*inputs, **kwargs)
        else: return self.averaged_model(*inputs, **kwargs)
    
    @torch.no_grad()
    def update_average(self):
        self.num_updates += 1
        if self.num_updates <= self.start_iteration:
            beta = 0.
        else:
            beta = self.beta
        source_dict = self.model.state_dict()
        target_dict = self.averaged_model.state_dict()
        for key in target_dict:
            target_dict[key].data.mul_(beta).add_(source_dict[key].data, alpha=1 - beta)

class Layer_worker(nn.Module):
    def __init__(self, normalization, in_dim, out_dim, bias=True):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim, bias)
        self.normalization = normalization
        if self.normalization: 
            if self.normalization == "spectral_norm":  
                self.layer = nn.utils.parametrizations.spectral_norm(self.layer)
                self.normalization = False
            elif self.normalization == "LayerNorm":
                 self.normalization = nn.LayerNorm(out_dim)
            elif self.normalization == "none":
                self.normalization = False
            else:
                pprint(NotImplemented)
                self.normalization = False
        else:
            self.normalization = False
        
    def forward(self, x):
        x = self.layer(x)
        if self.normalization: x = self.normalization(x)
        return x

class Layer(nn.Module):
    def __init__(self, normalization):
        super().__init__()    
        self.normalization = normalization
    
    def __getitem__(self, *args):
        return Layer_worker(self.normalization, *args)

class ResLayer(nn.Module):
    def __init__(self, input_dim, output_dim, layer, activation):
        super().__init__()
        self.layer_1 = layer.__getitem__(input_dim, input_dim)
        self.activation = activation
        self.layer_2 = layer.__getitem__(input_dim, output_dim)
    
    def forward(self, x):
        return x + self.layer_2(self.activation(self.layer_1(x))) 

class Layers(nn.Module):
    def __init__(self, n_layers, res_layers, input_dim, output_dim, activation, normalization, init_power_two, steady_dim):
        super().__init__()
        self.n_layers = n_layers
        self.res_layers = res_layers
        self.layer = Layer(normalization)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.init_power_two = init_power_two
        self.steady_dim = steady_dim
        
        self.layers, self.dims = self.get_architecture() 
        self.layer_dict = {0: {"layer": self.layer.__getitem__, "args": {}},
                           1: {"layer": ResLayer, "args": {"layer": self.layer, 
                                                           "activation": self.activation}}}

    def get_next_power_of_two(self, dim, factor):
        if factor == 2:
            return 2**list(filter(lambda x: (2**x >= dim), range(10)))[0]
        else:
            return 2**list(filter(lambda x: (2**x <= dim), range(10)))[-1]
    
    def get_operations(self):
        if self.input_dim < self.output_dim:
            operation, factor = min, 2
        elif self.input_dim > self.output_dim:
            operation, factor = max, 0.5
        else: 
            operation, factor = min, 1
        return operation, factor

    def get_dims(self):
        operation, factor = self.get_operations()
        dims = []
        step_dim = self.input_dim
        for i in range(self.n_layers):
            input_dim = step_dim
            if self.steady_dim: 
                if i == self.n_layers - 1: step_dim = self.output_dim    
            else:
                if i == 0 and self.init_power_two:
                    step_dim = operation(self.output_dim, self.get_next_power_of_two(input_dim, factor))
                else:
                    if self.res_layers <= i:
                        if i == self.n_layers - 1: 
                            step_dim = self.output_dim 
                        else:
                            if input_dim not in [2**i for i in range(10)]:
                                step_dim = operation(self.output_dim, self.get_next_power_of_two(input_dim, factor))
                            else: 
                                step_dim = operation(self.output_dim, input_dim * factor)
            dims.append([int(input_dim), int(step_dim)]) 
        assert self.output_dim == step_dim
        return dims
    
    def get_layers(self):
        layers = []
        for i in range(self.n_layers):
            if i == 0 and self.init_power_two:
                layers.append(0)
                self.res_layers += 1
            elif self.res_layers > i:
                layers.append(1)
            else:
                layers.append(0)
        return layers
    
    def get_architecture(self):
        assert not self.init_power_two or not self.steady_dim
        assert self.n_layers > self.res_layers
        layers = self.get_layers()
        dims = self.get_dims()
        return layers, dims

    def get_layer(self, i):
        layer = self.layer_dict[self.layers[i]]
        layer = layer["layer"](*self.dims[i], **layer["args"])
        return layer

class Pooling(pl.LightningModule):
    def __init__(self, normalization, pool, dim, n_variants):
        super().__init__()
        if pool not in ('sum', 'max', 'attention','softmax'):  raise ValueError(f'Unknown pooling operation {pool}')
        self.layer = Layer(normalization)
        self.pool = pool
        self.dim = dim
        self.n_variants = n_variants

        self.f, self.f_args = self.get_function()
            
    def get_function(self):
        if self.pool == "sum": return torch.sum, {"dim": 2} 
        elif self.pool == "attention":
            '''
                Modeled after Set Transformer (Lee et al., 2019)
                paper: http://proceedings.mlr.press/v97/lee19d.html
                original code: https://github.com/juho-lee/set_transformer/blob/master/models.py#L3 
            '''
            self.fc_q = self.layer.__getitem__(self.n_variants, 1)
            self.fc_k = self.layer.__getitem__(self.n_variants, 1)
            self.fc_v = self.layer.__getitem__(self.n_variants, 1)

            self.S = nn.Parameter(torch.Tensor(1, self.dim, self.n_variants))
            nn.init.xavier_uniform_(self.S)
            return  nn.MultiheadAttention(1, bias=False, num_heads=1, batch_first=True), {} 
        elif self.pool == 'softmax':
            '''
                Modeled after Enformer from DeepMind
                paper: https://www.nature.com/articles/s41592-021-01252-x 
                original code: https://github.com/deepmind/deepmind-research/blob/cb555c241b20c661a3e46e5d1eb722a0a8b0e8f4/enformer/enformer.py#L244  
                pytorch remade code: https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/modeling_enformer.py#L134 
            '''
            self.to_attn_logits = self.layer.__getitem__(self.n_variants, self.n_variants, False) #bias = False
            nn.init.eye_(self.to_attn_logits.layer.weight)
            self.gain = 2.0 #When 0.0 is equivalent to avg pooling, and when ~2.0 and `per_channel=False` it's equivalent to max pooling.
            with torch.no_grad(): self.to_attn_logits.layer.weight.mul_(self.gain)
            return torch.sum, {"dim": -1}  
        else: return torch.max, {"dim": 2} 

    def forward(self, x):
        if self.pool == "attention": 
            x = x.permute((0,2,1)) # make input x.shape = samples x phi_latent x variants --> to pool across variant dimension   
            #pad variant dim to max_num_variants across all the phenotypes
            if x.shape[-1] < self.n_variants: x = pad_variants(x,self.n_variants)  
            x, k, v = self.fc_q(self.S.repeat(x.size(0), 1, 1)), self.fc_k(x), self.fc_v(x)
            self.f_args = {"key": k, "value": v, "need_weights": False}
        if self.pool == "softmax":
            x = x.permute((0,1,3,2))
            if x.shape[-1] < self.n_variants: x = pad_variants(x,self.n_variants)  
            x = x.unsqueeze(3)  #Rearrange('b g (v p) l -> b g l v p', p = self.pool_size)
            x = x * self.to_attn_logits(x).softmax(dim=-1)
            
        x = self.f(x, **self.f_args)
        
        if self.pool == "attention": x = x[0].squeeze(2)
        if self.pool == "softmax": x = x.squeeze(-1)
        if self.pool == "max": x = x.values
        return x
