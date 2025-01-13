from typing import List
from numpy import ndarray
import numpy as np

from layers import Layer
from losses import Loss, MeanSquaredError


class LayerBlock(object):

    def __init__(self, layers: List[Layer]):
        super().__init__()
        self.layers = layers

    def forward(self,
                X_batch: ndarray,
                inference=False) ->  ndarray:

        X_out = X_batch
        for layer in self.layers:
            X_out = layer.forward(X_out, inference)

        return X_out
        
    def backward(self, loss_grad: ndarray) -> ndarray:

        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad

    def params(self):
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        for layer in self.layers:
            yield from layer.param_grads

    def __iter__(self):
        return iter(self.layers)

    def __repr__(self):
        layer_strs = [str(layer) for layer in self.layers]
        return f"{self.__class__.__name__}(\n  " + ",\n  ".join(layer_strs) + ")"


class NeuralNetwork(LayerBlock):
    '''
    Just a list of layers that runs forwards and backwards
    '''
    def __init__(self,
                 layers: List[Layer],
                 loss: Loss = MeanSquaredError,
                 seed: int = 1):
        super().__init__(layers)
        self.loss = loss
        self.seed = seed
        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)

    def forward_loss(self,
                     X_batch: ndarray,
                     y_batch: ndarray,
                     inference: bool = False) -> float:

        if X_batch is None:
            X_batch = self.layers[0].input_
        
        if y_batch is None:
            y_batch = self.loss.target

        prediction = self.forward(X_batch, inference)
        return self.loss.forward(prediction, y_batch) #normalized?

    
    def forward_loss_from_neuron(self, layerIdx, neuronIdx):
        # print('forward: layerIdx = ' + str(layerIdx) + '; neuronIdx = ' + str(neuronIdx))
        lay = self.layers[layerIdx]
        X_out = lay.input_

        # last_out = self.layers[-1].output
        # # lay.params[paramIdx][:, neuronIdx] += stepVector (should be in optimizer)
        # neuron_params = lay.params[0][:, neuronIdx]
        # neuron_out = np.dot(X_out, neuron_params)
        # lay.operations[0].output[:, neuronIdx] = neuron_out
        # lay.operations[1].input_[:, neuronIdx] = neuron_out
        # neuron_out += lay.params[1][0, neuronIdx]
        # lay.operations[1].output[:, neuronIdx] = neuron_out
        # act = lay.operations[2]
        # act.input_[:, neuronIdx] = neuron_out
        # neuron_out = act.just_activate(neuron_out)
        # act.output[:, neuronIdx] = neuron_out # need sometimes for calc grad
        # lay.output[:, neuronIdx] = neuron_out
        
        # if layerIdx == 1 and not np.allclose(last_out, lay.output):
        #     print('something was change after optimized forward')
        # print(np.allclose(X_out, lay_out))
        # X_out = lay.output
        # opt_out = lay.output
        for i in range(layerIdx, len(self.layers)):
            layer = self.layers[i]
            X_out = layer.forward(X_out)
        # if not np.allclose(X_out, last_out):
        #     print('something was change after classic forward')
        # assert np.allclose(opt_out, X_out), 'results divergate'
        # print(np.allclose(X_out, lay_out))
        return self.loss.forward(X_out, self.loss.target)

    def backward_loss_to_layer(self, loss_grad: ndarray, layerIdx, neuronIdx) -> ndarray:
        # print('backward_loss_to_neuron( lay = ' + str(layerIdx) + ', neu = ' + str(neuronIdx))
        # print('loss_grad:')
        # print(loss_grad)
        grad = loss_grad
        for i in range(len(self.layers) - 1, layerIdx - 1, -1):
            layer = self.layers[i]
            grad = layer.backward(grad)
        # print('grad of activate func:')
        # print(self.layers[layerIdx].operations[2].input_grad[:, neuronIdx])

        # print('grad of neu:')
        # print(grad[:, neuronIdx])
        # print()
        return layer.param_grads[0]

    def train_batch(self,
                    X_batch: ndarray,
                    y_batch: ndarray,
                    inference: bool = False) -> float:
        
        if X_batch is None:
            X_batch = self.layers[0].input_
        
        if y_batch is None:
            y_batch = self.loss.target

        prediction = self.forward(X_batch, inference)

        batch_loss = self.loss.forward(prediction, y_batch) #normalized?
        loss_grad = self.loss.backward()

        self.backward(loss_grad)

        return batch_loss
