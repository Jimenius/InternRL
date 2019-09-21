from torch import nn
from torch.nn import functional as F
import yaml

class TorchNet(nn.Module):
    '''
    Description:
        PyTorch model built from layer descriptions
    '''

    def __init__(self, layers):
        super(TorchNet, self).__init__()
        out_neurons = {}
        self.activations = {}
        self.layers = layers
        for layer in self.layers:
            ltype = layer['type']
            name = layer['name']
            prec = layer['preceding_layers']
            if ltype == 'Input':
                neurons = layer['shape'][0]
            elif ltype == 'Dense':
                self.activations[name] = layer['activation']
                neurons = layer['neurons']
                setattr(self, name, nn.Linear(out_neurons[prec[0]], neurons))
            else:
                raise ValueError('Layer not supported')
            out_neurons[name] = neurons

    def forward(self, x):
        activatation_function = {
            'relu': F.relu,
            'sigmoid': F.sigmoid,
            'tanh': F.tanh
        }
        tensors = {}
        outputs = []
        for layer in self.layers:
            name = layer['name']
            if layer['type'] == 'Input':
                tensors[name] = x
            prec = layer['preceding_layers']
            x = tensors[prec[0]]
            x = getattr(self, name)(x)
            if name in self.activations:
                x = activatation_function[self.activations[name]](x)
            tensors[name] = x
            if layer['output']:
                outputs.append(x)
        return outputs

def ModelBuilder(path):
    '''
    Description:
        Build model from the given path of the YAML file
    
    Inputs:
    path: str
        Path to the model description YAML file
    
    Outputs:
    Module
        PyTorch Network Module
    '''

    with open(path) as f:
        layers = yaml.load(f)

    model = TorchNet(layers)
    return model