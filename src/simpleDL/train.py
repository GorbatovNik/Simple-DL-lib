
from typing import Tuple
from copy import deepcopy

import numpy as np
from numpy import ndarray
import math

from simpleDL.network import NeuralNetwork
from simpleDL.optimizers import Optimizer
from simpleDL.utils.np_utils import permute_data

from tqdm import tqdm
import keyboard


class Trainer(object):
    '''
    Just a list of layers that runs forwards and backwards
    '''
    def __init__(self,
                 net: NeuralNetwork,
                 optim: Optimizer) -> None:
        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        setattr(self.optim, 'net', self.net)

    def fit(self, X_train: ndarray, y_train: ndarray,
            X_test: ndarray, y_test: ndarray,
            epochs: int=100,
            eval_every: int=10,
            batch_size: int=32,
            seed: int = 1,
            single_output: bool = False,
            restart: bool = True,
            early_stopping: bool = True,
            conv_testing: bool = False,
            permute_data: bool = True)-> None:

        setattr(self.optim, 'max_epochs', epochs)
        self.optim._setup_decay()

        np.random.seed(seed)
        if restart:
            for layer in self.net.layers:
                layer.first = True

            self.best_loss = 1e9

        for e in range(epochs):

            if (e+1) % eval_every == 0:

                last_model = deepcopy(self.net)

            if permute_data:
                X_train, y_train = permute_data(X_train, y_train)

            batch_generator = self.generate_batches(X_train, y_train,
                                                    batch_size)
            def check_for_q():
                return keyboard.is_pressed("q")
            for ii, (X_batch, y_batch) in tqdm(enumerate(batch_generator)):
                if check_for_q():
                    break
                self.net.train_batch(X_batch, y_batch)

                if type(self.optim) == 'ConjugateGradient': 
                    self.optim.step(X_batch, y_batch)
                else:
                    self.optim.step()
                # for li, layer in enumerate(self.net.layers):
                #     print('layer ' + str(li) + ':')
                #     for pi, param in enumerate(layer.params):
                #         par_len = math.sqrt(np.sum(param**2))
                #         print(' param ' + str(pi) + ' = ' + str(par_len))
                #         assert not np.isnan(par_len), "par_sum содержит NaN"

                # print('layer = ' + str(par_sum))

                if conv_testing:
                    if ii % 10 == 0:
                        test_preds = self.net.forward(X_batch, inference=True)
                        batch_loss = self.net.loss.forward(test_preds, y_batch)
                        print("batch",
                              ii,
                              "loss",
                              batch_loss)

                    if ii % 100 == 0 and ii > 0:
                        print("Validation accuracy after", ii, "batches is",
                        f'''{np.equal(np.argmax(self.net.forward(X_test, inference=True), axis=1),
                        np.argmax(y_test, axis=1)).sum() * 100.0 / X_test.shape[0]:.2f}%''')

            if (e+1) % eval_every == 0:

                test_preds = self.net.forward(X_test, inference=True)
                loss = self.net.loss.forward(test_preds, y_test)

                par_sum = math.sqrt(np.sum(self.net.layers[0].params[0]**2))
                print('par_sum = ' + str(par_sum))

                if early_stopping:
                    if loss < self.best_loss:
                        print(f"Validation loss after {e+1} epochs is {loss:.3f}")
                        self.best_loss = loss
                    else:
                        print()
                        print(f"Loss increased after epoch {e+1}, final loss was {self.best_loss:.3f},",
                              f"\nusing the model from epoch {e+1-eval_every}")
                        self.net = last_model
                        # ensure self.optim is still updating self.net
                        setattr(self.optim, 'net', self.net)
                        break
                else:
                    print(f"Validation loss after {e+1} epochs is {loss:.3f}")

            if self.optim.final_lr:
                self.optim._decay_lr()


    def generate_batches(self,
                         X: ndarray,
                         y: ndarray,
                         size: int = 32) -> Tuple[ndarray]:

        assert X.shape[0] == y.shape[0], \
        '''
        features and target must have the same number of rows, instead
        features has {0} and target has {1}
        '''.format(X.shape[0], y.shape[0])

        N = X.shape[0]

        for ii in range(0, N, size):
            X_batch, y_batch = X[ii:ii+size], y[ii:ii+size]

            yield X_batch, y_batch
