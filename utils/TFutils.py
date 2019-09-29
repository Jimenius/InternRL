import yaml
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, BatchNormalization, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *

def DenseLayer(x, units = 32, use_bn = True, activation = 'relu', drop_rate = 0., name = ''):
    '''
    Description:
        A fully-connected layer, with BatchNormalization and Dropout inserted if specified

    Inputs: 
    x: Tensor
        Input tensor
    units: int
        Output units
    use_bn: boolean
        Use Batch Normalization or not
    activation: str
        Activation function
    drop_rate: float 
        Fraction of the input units to drop, 0. for no dropout
    layer_num: str
        Current index of the layer to specify names of layers

    Outputs:
    Tensor
        Output of the layer
    '''

    y = Dense(units, name = name)(x)
    if use_bn:
        y = BatchNormalization(name = '{}_BN'.format(name))(y)
    if activation:
        y = Activation(activation, name = '{}_{}'.format(name, activation))(y)
    if drop_rate:
        y = Dropout(drop_rate, name = '{}_Drop'.format(name))(y)
    return y

def ModelBuilder(path):
    '''
    Description:
        Build model from the given path of the YAML file
    
    Inputs:
    path: str
        Path to the model description YAML file
    
    Outputs:
    Model
        Keras functional model
    '''

    # Parse YAML file as dict
    with open(path) as f:
        layers = yaml.load(f)

    inputs = [] # List of input layers
    outputs = [] # List of output layers
    layer_map = {} # Map layer name to its output tensor
    for layer in layers:
        ltype = layer['type']
        name = layer['name']
        prec = layer['preceding_layers']
        if ltype == 'Input':
            x = Input(shape = layer['shape'], name = name)
            inputs.append(x)
        elif ltype == 'Dense':
            activation = layer['activation']
            if activation:
                activation = activation.lower()
            x = DenseLayer(
                layer_map[prec[0]], units = layer['neurons'],
                use_bn = layer['bn'], activation = activation,
                drop_rate = layer['dropout'], name = name
                )
        elif ltype == 'Concatenate':
            tensors = map(lambda layer_name: layer_map[layer_name], prec)
            x = Concatenate()(list(tensors))
        elif ltype == 'Lambda':
            x = Lambda(eval(layer['FUNC']))(layer_map[prec[0]])
        layer_map[name] = x
        if layer['output']:
            outputs.append(x)

    model = Model(inputs = inputs, outputs = outputs)
    model.summary()
    return model

def get_optimizer(name = 'Adam', learning_rate = 0.001):
    '''
    Description:
        Get optimizer from name
    
    Inputs:
    name: str
        Optimizer name
    lr: float
        Learning rate
    
    Outputs:
    Optimizer
        Keras optimizer
    '''

    name = name.lower()
    optimizers = {
        'sgd': SGD,
        'adam': Adam,
        'rmsprop': RMSprop
    }
    return optimizers[name](learning_rate = learning_rate)