# SimpleDL

A modular library for defining, training, and evaluating neural networks. It supports various layers, activation functions, loss functions, and optimizers.

## Features

- **Layers**: Fully connected (Dense) layers.
- **Activations**: ReLU, Sigmoid, Tanh, Linear.
- **Loss Functions**: Softmax Cross-Entropy, Mean Squared Error (MSE).
- **Optimizers**: SGD, SGD with Momentum, Conjugate Gradient, Fletcher-Reeves, BFGS, BFGD-L.
- **Trainer**: Simplifies model training with configurable parameters.
- **Dataset**: MNIST support with preprocessing utilities (normalization, one-hot encoding).

## Instalation
- Ensure you are in the root directory, where the pyproject.toml file is located.
- Run:

1) ```python -m build```

2) ```pip install dist/simpleDL-0.1.0-py3-none-any.whl```
