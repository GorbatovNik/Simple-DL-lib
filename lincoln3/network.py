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

        prediction = self.forward(X_batch, inference)
        return self.loss.forward(prediction, y_batch) #normalized?

    
    def forward_loss_from_neuron(self, layerIdx, neuronIdx):
        lay = self.layers[layerIdx]
        X_out = lay.input_
        # lay.params[paramIdx][:, neuronIdx] += stepVector (should be in optimizer)
        neuron_params = lay.params[0][:, neuronIdx]
        neuron_out = np.dot(X_out, neuron_params)
        lay.operations[0].output[:, neuronIdx] = neuron_out
        lay.operations[1].input_[:, neuronIdx] = neuron_out
        neuron_out += lay.params[1][0, neuronIdx]
        lay.operations[1].output[:, neuronIdx] = neuron_out
        act = lay.operations[2]
        act.input_[:, neuronIdx] = neuron_out
        neuron_out = act.just_activate(neuron_out)
        act.output[:, neuronIdx] = neuron_out # need sometimes for calc grad
        lay.output[:, neuronIdx] = neuron_out

        # X_out = lay.output
        for i in range(layerIdx, len(self.layers)):
            layer = self.layers[i]
            X_out = layer.forward(X_out)

        return self.loss.forward(X_out, self.loss.target)

    def backward_loss_to_neuron(self, loss_grad: ndarray, layerIdx, neuronIdx) -> ndarray:
        print('backward_loss_to_neuron( lay = ' + str(layerIdx) + ', neu = ' + str(neuronIdx))
        print('loss_grad:')
        print(loss_grad)
        grad = loss_grad
        for i in range(len(self.layers) - 1, layerIdx+1, -1):
            layer = self.layers[i]
            grad = layer.backward(grad, skeep_param_grads=True)
        if len(self.layers) > layerIdx + 1:
            prev_lay = self.layers[layerIdx + 1]
            grad = prev_lay.operations[2].backward(grad)
            grad = prev_lay.operations[1].backward(grad, skeep_param_grads=True)
            grad = np.dot(grad, prev_lay.params[0][neuronIdx, :])
            prev_lay.operations[0].input_grad[:, neuronIdx] = None
            prev_lay.operations[0].param_grad = None
        else:
            grad = loss_grad[:, neuronIdx]

        lay = self.layers[layerIdx]
        grad = lay.operations[2].just_grad(grad, lay.operations[2].output[:, neuronIdx])
        print('grad of activate func:')
        print(grad)
        lay.operations[1].input_grad[:, neuronIdx] = None
        lay.operations[1].param_grad[:, neuronIdx] = None
        grad = np.dot(lay.input_.transpose(1, 0), grad)
        print('grad of neu:')
        print(grad)
        print()

        return grad

    def train_batch(self,
                    X_batch: ndarray,
                    y_batch: ndarray,
                    inference: bool = False) -> float:

        prediction = self.forward(X_batch, inference)

        batch_loss = self.loss.forward(prediction, y_batch) #normalized?
        loss_grad = self.loss.backward()

        self.backward(loss_grad)

        return batch_loss
