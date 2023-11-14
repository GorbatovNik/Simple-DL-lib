import numpy as np
from scipy.optimize import line_search
import warnings
warnings.filterwarnings("ignore")

class Optimizer(object):
    def __init__(self,
                 lr: float = 0.01,
                 final_lr: float = 0,
                 decay_type: str = None) -> None:
        self.lr = lr
        self.final_lr = final_lr
        self.decay_type = decay_type
        self.first = True

    def _setup_decay(self) -> None:

        if not self.decay_type:
            return
        elif self.decay_type == 'exponential':
            self.decay_per_epoch = np.power(self.final_lr / self.lr,
                                       1.0 / (self.max_epochs - 1))
        elif self.decay_type == 'linear':
            self.decay_per_epoch = (self.lr - self.final_lr) / (self.max_epochs - 1)

    def _decay_lr(self) -> None:

        if not self.decay_type:
            return

        if self.decay_type == 'exponential':
            self.lr *= self.decay_per_epoch

        elif self.decay_type == 'linear':
            self.lr -= self.decay_per_epoch

    def step(self,
             epoch: int = 0) -> None:

        for (param, param_grad) in zip(self.net.params(),
                                       self.net.param_grads()):
            self._update_rule(param=param,
                              grad=param_grad)

    def _update_rule(self, **kwargs) -> None:
        raise NotImplementedError()

class GD(Optimizer):
    pass

class SGD(Optimizer):
    def __init__(self,
                 lr: float = 0.01,
                 final_lr: float = 0,
                 decay_type: str = None) -> None:
        super().__init__(lr, final_lr, decay_type)

    def _update_rule(self, **kwargs) -> None:

        update = self.lr*kwargs['grad']
        kwargs['param'] -= update

class SGDMomentum(Optimizer):
    def __init__(self,
                 lr: float = 0.01,
                 final_lr: float = 0,
                 decay_type: str = None,
                 momentum: float = 0.9) -> None:
        super().__init__(lr, final_lr, decay_type)
        self.momentum = momentum

    def step(self) -> None:
        if self.first:
            self.velocities = [np.zeros_like(param)
                               for param in self.net.params()]
            self.first = False

        for (param, param_grad, velocity) in zip(self.net.params(),
                                                 self.net.param_grads(),
                                                 self.velocities):
            self._update_rule(param=param,
                              grad=param_grad,
                              velocity=velocity)

    def _update_rule(self, **kwargs) -> None:

            # Update velocity
            kwargs['velocity'] *= self.momentum
            kwargs['velocity'] += self.lr * kwargs['grad']

            # Use this to update parameters
            kwargs['param'] -= kwargs['velocity']


class AdaGrad(Optimizer):
    def __init__(self,
                 lr: float = 0.01,
                 final_lr_exp: float = 0,
                 final_lr_linear: float = 0) -> None:
        super().__init__(lr, final_lr_exp, final_lr_linear)
        self.eps = 1e-7

    def step(self) -> None:
        if self.first:
            self.sum_squares = [np.zeros_like(param)
                                for param in self.net.params()]
            self.first = False

        for (param, param_grad, sum_square) in zip(self.net.params(),
                                                   self.net.param_grads(),
                                                   self.sum_squares):
            self._update_rule(param=param,
                              grad=param_grad,
                              sum_square=sum_square)

    def _update_rule(self, **kwargs) -> None:

            # Update running sum of squares
            kwargs['sum_square'] += (self.eps +
                                     np.power(kwargs['grad'], 2))

            # Scale learning rate by running sum of squareds=5
            lr = np.divide(self.lr, np.sqrt(kwargs['sum_square']))

            # Use this to update parameters
            kwargs['param'] -= lr * kwargs['grad']


class RegularizedSGD(Optimizer):
    def __init__(self,
                 lr: float = 0.01,
                 alpha: float = 0.1) -> None:
        super().__init__()
        self.lr = lr
        self.alpha = alpha

    def step(self) -> None:

        for (param, param_grad) in zip(self.net.params(),
                                       self.net.param_grads()):

            self._update_rule(param=param,
                              grad=param_grad)

    def _update_rule(self, **kwargs) -> None:

            # Use this to update parameters
            kwargs['param'] -= (
                self.lr * kwargs['grad'] + self.alpha * kwargs['param'])

class ConjugateGradient(Optimizer):
    def __init__(self,
                 lr: float = 0.01) -> None:
        super().__init__(lr)

    def goldsect(self, l=0, r=10.0):
        # f_l = f(x, w0, w1, targ)
        
        m1 = l + (r - l)*0.38
        m2 = l + (r - l)*0.62

        # f_m1 = f(x, w0 + d0*m1, w1 + d1*m1, targ)
        # f_m2 = f(x, w0 + d0*m2, w1 + d1*m2, targ)
        for (param, param_grad) in zip(self.net.params(),
                                       self.net.param_grads()):
            param -= param_grad*m1
    
        f_m1 = self.net.forward_loss()

        for (param, param_grad) in zip(self.net.params(),
                                       self.net.param_grads()):
            param -= param_grad*(m2-m1)
        
        f_m2 = self.net.forward_loss()

        for (param, param_grad) in zip(self.net.params(),
                                       self.net.param_grads()):
            param -= -param_grad*m2
        
        while r - l > 0.001:
            if f_m1 < f_m2+ 0.0000000000001:
                r = m2
            else:
                l = m1
            
            m1 = l + (r - l)*0.38
            m2 = l + (r - l)*0.62

            for (param, param_grad) in zip(self.net.params(),
                                       self.net.param_grads()):
                param -= param_grad*m1
    
            f_m1 = self.net.forward_loss()

            for (param, param_grad) in zip(self.net.params(),
                                        self.net.param_grads()):
                param -= param_grad*(m2-m1)
            
            f_m2 = self.net.forward_loss()

            for (param, param_grad) in zip(self.net.params(),
                                        self.net.param_grads()):
                param -= -param_grad*m2
        return l
    
    def step(self) -> None:

        layerIdx, neuronIdx = None, None

        def obj_func(new_params):
            nonlocal layerIdx, neuronIdx
            if neuronIdx < 0: # bias
                self.net.layers[layerIdx].params[1][1, :] = new_params
            else: # weights
                self.net.layers[layerIdx].params[0][:, neuronIdx] = new_params
                loss = self.net.forward_loss_from_neuron(layerIdx, neuronIdx)
                # print('obj_func returns: ' + str(loss))
            return loss

        def obj_grad(new_params):
            nonlocal layerIdx, neuronIdx
            if neuronIdx < 0: # bias
                self.net.layers[layerIdx].params[1][1, :] = new_params
            else: # weights
                self.net.layers[layerIdx].params[0][:, neuronIdx] = new_params
        
            # batch_loss = self.net.loss.target #normalized?
            self.net.forward_loss_from_neuron(layerIdx, neuronIdx)
            loss_grad = self.net.loss.backward()

            return self.net.backward_loss_to_neuron(loss_grad, layerIdx, neuronIdx)

        weight_lams = [[0.0 for neuIdx in range(lay.params[0].shape[1])] for _, lay in enumerate(self.net.layers)]
        # bias_lams = [0.0 for _, lay in enumerate(self.net.layers)]

        for layIdx, lay in enumerate(self.net.layers[::-1]):
            layIdx = len(self.net.layers) - layIdx - 1
            for neuIdx in range(lay.params[0].shape[1]):
                layerIdx = layIdx
                neuronIdx = neuIdx
                start_point = lay.params[0][:, neuIdx]
                # print(type(lay.param_grads[0]))
                # assert type(lay.param_grads[0]) == 'WeightMultiply'
                search_antigradient = -lay.param_grads[0][:, neuIdx]
                lam, iter, grad_calcs, new_fval, old_fval, _ = line_search(obj_func, obj_grad, start_point, search_antigradient, c2=0.1)
                if lam is None:
                    continue
                win = - new_fval + old_fval
                # print('win = ' + str(win))
                cit = 0
                while win > 0.01:
                    cit += 1
                    start_point = start_point + search_antigradient * lam
                    self.net.layers[layerIdx].params[0][:, neuronIdx] = start_point
                    print('forward loss = ' + str(self.net.forward_loss_from_neuron(layerIdx, neuronIdx)))
                    loss_grad = self.net.loss.backward()
                    grad = self.net.backward_loss_to_neuron(loss_grad, layerIdx, neuronIdx)
                    search_antigradient = -lay.param_grads[0][:, neuIdx]
                    are_equal = np.allclose(search_antigradient, -grad, atol=1e-15)
                    lam, iter, _, new_fval, old_fval, _ = line_search(obj_func, obj_grad, start_point, search_antigradient, c2=0.1)
                    if lam is None:
                        break;
                    win = - new_fval + old_fval
                    # print('win = ' + str(win))
                self.net.layers[layerIdx].params[0][:, neuronIdx] = start_point if lam is None else start_point + search_antigradient * lam
                if cit > 10:
                    print(cit)

        for layIdx, lay in enumerate(self.net.layers):
            # print(type(lay.param_grads[1]))
            # assert type(lay.param_grads[1]) == 'BiasAdd'
            lay.params[1] -= lay.param_grads[1] * self.lr
            # lay.params[0] -= lay.param_grads[0]*self.lr
