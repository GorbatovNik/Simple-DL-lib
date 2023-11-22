import numpy as np
from scipy.optimize import line_search
import warnings
import random
import math
from collections import deque
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


class BFGS(Optimizer):
    def __init__(self,
                 lr: float = 0.01,
                 depth: int = 5,
                 mini_epochs: int = 10) -> None:
        self.depth = depth
        self._lam_is_None_cnt = 0
        self.mini_epochs = mini_epochs
        super().__init__(lr)

    def step(self) -> None:
        # start_point = None

        def net_params_to_vector():
            result = np.array([])
            for lay in self.net.layers:
                result = np.append(result, np.append(lay.params[0], lay.params[1], axis=0).reshape(-1))
            return result
            
        def net_param_grads_to_vector():
            result = np.array([])
            for lay in self.net.layers:
                result = np.append(result, np.append(lay.param_grads[0], lay.param_grads[1], axis=0).reshape(-1))
            return result

        def apply_vector_of_params_to_net(params):
            offset = 0
            for lay in self.net.layers:
                new_mul = params[offset:offset+lay.params[0].size].reshape(lay.params[0].shape)
                offset+=lay.params[0].size
                lay.params[0] = new_mul
                lay.operations[0].param = lay.params[0]
                new_bias = params[offset:offset+lay.params[1].size].reshape(lay.params[1].shape)
                offset+=lay.params[1].size
                lay.params[1] = new_bias
                lay.operations[1].param = lay.params[1]

        # def _get_chain():
        def obj_func(new_params):
            # nonlocal layerIdx #, neuronIdx, start_point, search_antigradient
            # lay = self.net.layers[layerIdx]
            # flat_params = new_params.reshape(lay.params[0].shape)
            # lay.params[0] = flat_params
            # lay.operations[0].param = lay.params[0]
            apply_vector_of_params_to_net(new_params)
            loss = self.net.forward_loss(None, None)

            return loss

        def obj_grad(new_params):
            # nonlocal layerIdx
            # lay = self.net.layers[layerIdx]
            # flat_params = new_params.reshape(lay.params[0].shape)
            # lay.params[0] = flat_params
            # lay.operations[0].param = lay.params[0]
            apply_vector_of_params_to_net(new_params)
            loss = self.net.train_batch(None, None)
            grad = net_param_grads_to_vector()

            return grad

        if self.first:
            # first_lay_param_grads = self.net.layers[1].param_grads[0].reshape(-1).copy()
            self.bfgs =  {'s': deque(maxlen=self.depth),
                          'y': deque(maxlen=self.depth),
                          'ro': deque(maxlen=self.depth),
                          'prev_x': net_params_to_vector(),
                          'prev_g': net_param_grads_to_vector(),
                          'z': -net_param_grads_to_vector()}
            self.first = False
            # assert np.allclose(first_lay_param_grads, self.net.layers[1].param_grads[0].reshape(-1))
        else:
            # for layIdx, lay in enumerate(self.net.layers):
            bfgs = self.bfgs
            q = bfgs['prev_g']
            actual_depth = len(bfgs['s'])
            print('actual_depth = ' + str(actual_depth))
            alphas = []
            for ro, s, y in zip(bfgs['ro'], bfgs['s'], bfgs['y']):
                alpha = ro * np.dot(s, q)
                q = q - alpha*y
                alphas.append(alpha)
            
            gamma = 1
            if actual_depth > 0:
                gamma = np.dot(bfgs['s'][0], bfgs['y'][0])/np.dot(bfgs['y'][0], bfgs['y'][0])
            
            z = gamma*q
            zp = list(zip(alphas, bfgs['ro'], bfgs['s'], bfgs['y']))
            for alpha, ro, s, y in reversed(zp):
                beta = ro*np.dot(y, z)
                z = z + s*(alpha - beta)

            # y = lay.param_grads[0].reshape(-1) - bfgs['prev_g']
            # s = lay.params[0].reshape(-1) - bfgs['prev_x']
            # ro = 1/np.dot(y, s)
            # if ro >= 0 or ro < 10:
            bfgs['z'] = -z #/np.linalg.norm(z)
            params_vector = net_params_to_vector()
            bfgs['s'].appendleft(params_vector - bfgs['prev_x'])
            bfgs['prev_x'] = params_vector
            param_grads_vector = net_param_grads_to_vector()
            bfgs['y'].appendleft(param_grads_vector - bfgs['prev_g'])
            bfgs['prev_g'] = param_grads_vector
            ys = np.dot(bfgs['y'][0], bfgs['s'][0])
            print('ys = ', str(ys))
            # assert ys > 0, 'Curvature condition was not satisfy'
            # assert ys > 1e-5, 'transpose(y)*s too close to zero'
            bfgs['ro'].appendleft(1/np.dot(bfgs['y'][0], bfgs['s'][0]))
            z_norm = np.sqrt(np.sum(z**2))
            print('z norm = ' + str(z_norm))
            assert not math.isnan(z_norm)
            # else:
            #     print('ro dropped')

        # for layIdx, lay in enumerate(self.net.layers):
        start_point = net_params_to_vector()
        search_direction = self.bfgs['z'].copy()
        # search_direction /= np.linalg.norm(search_direction)
        # st_was = start_point.copy()
        # layerIdx = layIdx
        # print('step size: ' + str(np.linalg.norm(lay.param_grads[0].reshape(-1))))
        # print('descent: ' + str(obj_func(start_point) - obj_func(start_point - lay.param_grads[0].reshape(-1))))

        # obj_grad(start_point)
        # # if lam is None:
        # grad = net_param_grads_to_vector()
        # grad *= 0.01
        # print('grad step size: ' + str(np.linalg.norm(grad)))
        # grad_win = obj_func(start_point) - obj_func(start_point - grad)
        # print('grad descent: ' + str(grad_win))
        # if grad_win < 0.0:
        #     assert False, 'grad_win = ' + str(grad_win)

        lam, iter, grad_calcs, new_fval, old_fval, _ = line_search(obj_func, obj_grad, start_point, search_direction, c1=1e-4, c2=0.9)

        if lam is None:
            self._lam_is_None_cnt += 1
            # search_direction = -search_direction
            print('Warning: lam is None. Cnt = ' + str(self._lam_is_None_cnt))
            obj_grad(start_point)
            search_direction = -net_param_grads_to_vector()

            lam, iter, grad_calcs, new_fval, old_fval, _ = line_search(obj_func, obj_grad, start_point, search_direction, c1=1e-4, c2=0.9)

        if lam is None:
            print('\nALERT!: lam is still None!')
            obj_grad(start_point)
            search_direction = -0.01*net_param_grads_to_vector()

            lam, iter, grad_calcs, new_fval, old_fval, _ = line_search(obj_func, obj_grad, start_point, search_direction, c1=1e-66, c2=0.999999)

        if lam is None:
            pass
            # print('grad_win was = ' + str(grad_win))
        assert lam is not None, 'lam is still None, WTF?'
        # assert(np.allclose(start_point, st_was))

        # print('dir step size: ' + str(np.linalg.norm(search_direction)))
        # print('dir descent: ' + str(obj_func(start_point) - obj_func(start_point + lam*search_direction)))
        # print('(dir, grad) = ' + str(np.dot(-grad/np.linalg.norm(grad), search_direction/np.linalg.norm(search_direction))))
        new_params = (start_point + lam*search_direction)
        apply_vector_of_params_to_net(new_params)
        obj_grad(new_params)
        # print('lay = 0 params[0] norm = ' + str(np.sqrt(np.sum(self.net.layers[0].params[0]**2))))
        # print('lay = 0 params[1] norm = ' + str(np.sqrt(np.sum(self.net.layers[0].params[1]**2))))
        # print('lay = 1 params[0] norm = ' + str(np.sqrt(np.sum(self.net.layers[1].params[0]**2))))
        # print('lay = 1 params[1] norm = ' + str(np.sqrt(np.sum(self.net.layers[1].params[1]**2))))
        print('-----------------------------')

class FletcherRevees(Optimizer):
    def __init__(self,
                 lr: float = 0.01) -> None:
        super().__init__(lr)

    def step(self) -> None:
        if self.first:
            self.prev_grads = [lay.params[0] for _, lay in enumerate(self.net.layers)]
            self.last_directions = [-lay.params[0] for _, lay in enumerate(self.net.layers)]
            self.first = False
        else:
            for i, lay in enumerate(self.net.layers):
                param_grad = lay.param_grads[0]
                prev_grad = self.prev_grads[i]
                numerators = np.diagonal(np.dot(param_grad.T, param_grad))
                denumerators = np.diagonal(np.dot(prev_grad.T, prev_grad))
                betas = numerators / denumerators
                for k in range(self.last_directions[i].shape[0]):
                    self.last_directions[i][k, :] *= betas
                self.last_directions[i] -= param_grad
            
            self.prev_grads = [lay.params[0] for _, lay in enumerate(self.net.layers)]
        
        for lay, direct in zip(self.net.layers, self.last_directions):
            lay.params[0] += self.lr*direct
            lay.params[1] -= self.lr*lay.param_grads[1]

class ConjugateGradient(Optimizer):
    def __init__(self,
                 lr: float = 0.01) -> None:
        super().__init__(lr)
    '''
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
    '''
    def step(self) -> None:

        layerIdx, neuronIdx = None, None
        start_point = None
        search_antigradient = None


        def obj_func(new_params):
            nonlocal layerIdx, neuronIdx, start_point, search_antigradient
            step = new_params - start_point
            if not np.allclose(step, np.zeros(step.shape)):
                step_norm, search_antigradient_norm = step/np.linalg.norm(step), search_antigradient/np.linalg.norm(search_antigradient)
                # print('step: ' + str(step_norm))
                # print('search_antigradient: ' + str(search_antigradient_norm))
                # assert np.allclose(step_norm, search_antigradient_norm), 'obj_func: non collinear'
                # print('obj_func: k = ' + str(np.linalg.norm(step-search_antigradient)))
            # else:
                # print('obj_func: k = 0')
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

            return self.net.backward_loss_to_layer(loss_grad, layerIdx, neuronIdx)[:, neuronIdx]

        # weight_lams = [[0.0 for neuIdx in range(lay.params[0].shape[1])] for _, lay in enumerate(self.net.layers)]
        # bias_lams = [0.0 for _, lay in enumerate(self.net.layers)]

        for layIdx, lay in enumerate(self.net.layers[::-1]):
            layIdx = len(self.net.layers) - layIdx - 1
            for neuIdx in range(lay.params[0].shape[1]):
                # if neuIdx == 5:
                    # print('that is')
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
                start_point = start_point + search_antigradient * lam
                while win > 0.01:
                    cit += 1
                    self.net.layers[layerIdx].params[0][:, neuronIdx] = start_point
                    # print('forward loss = ' + str(self.net.forward_loss_from_neuron(layerIdx, neuronIdx)))
                    loss_grad = self.net.loss.backward()
                    grad = self.net.backward_loss_to_layer(loss_grad, layerIdx, neuronIdx)[:, neuronIdx]
                    search_antigradient = -lay.param_grads[0][:, neuIdx]
                    are_equal = np.allclose(search_antigradient, -grad, atol=1e-15)
                    lam, iter, _, new_fval, old_fval, _ = line_search(obj_func, obj_grad, start_point, search_antigradient, c2=0.1)
                    if lam is None:
                        break;
                    win = - new_fval + old_fval
                    
                    start_point = start_point + search_antigradient * lam
                    # print('win = ' + str(win))
                self.net.layers[layerIdx].params[0][:, neuronIdx] = start_point
                if cit > 10:
                    print(cit)

        for layIdx, lay in enumerate(self.net.layers):
            # print(type(lay.param_grads[1]))
            # assert type(lay.param_grads[1]) == 'BiasAdd'
            lay.params[1] -= lay.param_grads[1] * self.lr
            # lay.params[0] -= lay.param_grads[0]*self.lr
