import numpy as np
from numpy import ndarray
from simpleDL.base import Operation


class Linear(Operation):
    '''
    Linear activation function
    '''
    def __init__(self) -> None:
        super().__init__()

    def just_activate(self, inp):
        return inp

    def _output(self, inference: bool) -> ndarray:
        return self.input_

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad
    
    def just_grad(self, out_grad, out):
        return out_grad


class Sigmoid(Operation):
    '''
    Sigmoid activation function
    '''
    def __init__(self) -> None:
        super().__init__()

    def just_activate(self, inp):
        np.clip(inp, -100, 100)
        return 1.0/(1.0+np.exp(-1.0 * inp))

    def _output(self, inference: bool) -> ndarray:
        # if np.max(np.abs(self.input_)) > 100:
            # print("acitvations _output(): |value| > 100")
        #     print('max = ' + str(np.max(np.abs(self.input_))))
        #     print(self.input_)
        np.clip(self.input_, -100, 100)
        return 1.0/(1.0+np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad
    
    def just_grad(self, out_grad, out):
        sigmoid_backward = out * (1.0 - out)
        inp_grad = sigmoid_backward * out_grad
        return inp_grad



class Tanh(Operation):
    '''
    Hyperbolic tangent activation function
    '''
    def __init__(self) -> None:
        super().__init__()

    def just_activate(self, inp):
        np.clip(inp, -100, 100)
        return np.tanh(inp)

    def _output(self, inference: bool) -> ndarray:
        np.clip(self.input_, -100, 100)
        return np.tanh(self.input_)

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        return output_grad * (1 - self.output * self.output)

    def just_grad(self, out_grad, out):
        return out_grad * (1 - out * out)

class ReLU(Operation):
    '''
    Hyperbolic tangent activation function
    '''
    def __init__(self) -> None:
        super().__init__()

    def just_activate(self, inp):
        return np.clip(inp, 0, None)

    def _output(self, inference: bool) -> ndarray:
        return np.clip(self.input_, 0, None)

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        mask = self.output >= 0
        return output_grad * mask

    def just_grad(self, out_grad, out):
        mask = out >= 0
        return out_grad * mask