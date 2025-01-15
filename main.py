import numpy as np
from mnist import MNIST
from simpleDL.layers import Dense
from simpleDL.losses import SoftmaxCrossEntropy, MeanSquaredError
from simpleDL.optimizers import Optimizer, SGD, SGDMomentum, ConjugateGradient, FletcherRevees, BFGS
from simpleDL.activations import Sigmoid, Tanh, Linear, ReLU
from simpleDL.network import NeuralNetwork
from simpleDL.train import Trainer
# from utils import mnist
from simpleDL.utils.np_utils import softmax

data_dir = "mnist_data"
mnist = MNIST(data_dir)
X_train, y_train = mnist.load_training()
X_test, y_test = mnist.load_testing()

num_labels = len(y_train)

# one-hot encode
num_labels = len(y_train)
train_labels = np.zeros((num_labels, 10))
for i in range(num_labels):
    train_labels[i][y_train[i]] = 1

num_labels = len(y_test)
test_labels = np.zeros((num_labels, 10))
for i in range(num_labels):
    test_labels[i][y_test[i]] = 1
    
X_train, X_test = X_train - np.mean(X_train), X_test - np.mean(X_train)
X_train, X_test = X_train / np.std(X_train), X_test / np.std(X_train)

print(X_train.shape)

def calc_accuracy_model(model, test_set):
    return print(f'''The model validation accuracy is: {np.equal(np.argmax(model.forward(test_set, inference=True), axis=1), y_test).sum() * 100.0 / test_set.shape[0]:.2f}%''')

model = NeuralNetwork(
    layers=[Dense(neurons=89, 
                  activation=Tanh(),
                  weight_init="glorot"),
            Dense(neurons=10, 
                  activation=Sigmoid(),
                  weight_init="glorot")],
            loss = MeanSquaredError(normalize=True), 
seed=20190119)

trainer = Trainer(model, SGDMomentum())
trainer.fit(X_train, train_labels, X_test, test_labels,
            epochs = 20,
            eval_every = 1,
            seed=20190119,
            batch_size=60000,
            permute_data=False);
print()
calc_accuracy_model(model, X_test)
