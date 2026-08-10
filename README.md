# NeuralTrainer — Neural Networks from Scratch with Live Visualization

An educational neural-network trainer implemented with **NumPy, Tkinter and Matplotlib**.

The project builds a fully connected feed-forward neural network without TensorFlow or PyTorch, implements explicit forward/backward propagation, supports configurable hidden layers and loss functions, and provides a desktop GUI for editing architecture, training data, learning rate, activation functions, weights and biases.

Its main goal is not benchmark performance. It is to make the mechanics of a neural network visible and editable while it trains.

## Highlights

- Dense neural network implemented directly in NumPy
- Explicit forward propagation
- Explicit backpropagation and gradient descent
- Configurable number and size of hidden layers
- Per-hidden-layer activation selection
- Multiple loss functions
- Gradient clipping
- Random or manually entered parameters
- Live network topology visualization
- Weight, bias and neuron-value visualization
- Live response curves through every layer
- Loss-vs-iteration plotting
- Tkinter desktop interface

## Supported activations

Hidden layers can use:

- ReLU
- Leaky ReLU
- Sigmoid
- Tanh
- Softplus
- Softmax (educational/experimental)

The current output layer is intentionally linear / identity, which makes the application most directly suited to small regression and function-approximation experiments.

## Supported losses

- Mean Squared Error (MSE)
- Mean Error (ME)
- Manhattan / L1 error
- Log error
- Log-likelihood-style loss

Some combinations are included for experimentation rather than as recommended mathematically paired activation/loss designs.

## How it works

For each layer, the network performs

```text
Z[l] = W[l] A[l-1] + b[l]
A[l] = activation(Z[l])
```

The output layer uses the identity activation.

During backpropagation, the selected loss derivative is propagated backward through every layer. Weight and bias gradients are computed explicitly with NumPy matrix operations and then updated with gradient descent.

Before each parameter update, gradients are clipped to reduce the chance of unstable exploding updates during interactive experiments.

## GUI

The Tkinter interface allows the user to configure the network without changing the source code.

### Network configuration

Users can choose:

- number of inputs;
- number of outputs;
- number of hidden layers;
- neuron count for every hidden layer;
- activation function for every hidden layer;
- learning rate;
- training iterations;
- loss function.

### Training sample

The GUI dynamically creates input and target fields based on the configured input/output dimensions.

The recovered version trains on the values entered in this panel as a single column sample. The NumPy network class itself accepts matrices with multiple sample columns and can therefore be used independently of the GUI for batch experiments.

### Manual parameters

Random initialization can be disabled. The interface then generates matrix-style controls for every weight and bias in the network.

This allows a user to inspect and deliberately set the exact parameters of a small neural network, which is useful when learning how forward propagation and backpropagation behave.

## Visualization

### Network topology

A live diagram shows:

- every layer;
- neurons;
- connections;
- current weights;
- bias values;
- neuron activations for the active input;
- current training iteration.

### Network response

The application can sweep the first input across a configurable numerical range while holding the remaining inputs at zero.

It then plots the response of every neuron in every layer. This makes it possible to watch how the learned function changes during optimization rather than inspecting only the final prediction.

### Training loss

After training, the loss history is plotted over all iterations.

## Repository structure

```text
neuralTrainer/
├── neural_trainer.py    # NumPy neural network + Tkinter GUI
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

## Installation

Python 3 is required. Tkinter is included with many standard Python distributions, although some Linux distributions package it separately.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it and install the numerical/plotting dependencies:

```bash
pip install -r requirements.txt
```

## Run the application

```bash
python neural_trainer.py
```

A Tkinter window will open with the network configuration controls.

## Programmatic use

The `NeuralNetwork` class can also be used without the GUI:

```python
import numpy as np
from neural_trainer import NeuralNetwork

X = np.array([[1.0], [2.0], [3.0]])
Y = np.array([[6.0]])

nn = NeuralNetwork(
    input_size=3,
    hidden_layers=[4, 4],
    output_size=1,
    activations=["leaky_relu", "tanh"],
    error_name="mse",
    learning_rate=0.0005,
    seed=42,
)

losses = nn.train(X, Y, iterations=1000)
prediction = nn.predict(X)

print(prediction)
```

## Recovered-project cleanup

This repository was reconstructed from an earlier version of the project that had not been committed to GitHub. The restored implementation preserves the original architecture and GUI concept while correcting copy/paste indentation damage and several small robustness issues.

Notable cleanup includes:

- valid Python module structure and entry point;
- numerically safer sigmoid and softplus implementations;
- shape validation for manually supplied parameters;
- corrected derivative sign for mean error;
- clearer separation between hidden activations and the identity output;
- safer visualization update intervals;
- corrected input-layer labeling in response plots;
- reusable `predict()` method;
- optional deterministic seed.

## Mathematical caveats

The project is intentionally educational, so several design choices should be interpreted accordingly.

- The softmax derivative helper is only an element-wise approximation; the full derivative is a Jacobian.
- Softmax is therefore not intended here as a mathematically complete hidden-layer backpropagation implementation.
- Log-likelihood is normally paired with a probability-producing output activation, whereas this implementation uses a linear output layer.
- Initial weights are sampled from a wide uniform interval to preserve the behaviour of the recovered project rather than using Xavier/He initialization.
- Training uses straightforward full-batch gradient descent rather than modern optimizers.

These are useful directions for extension rather than hidden implementation details.

## Possible extensions

- Xavier and He initialization
- categorical softmax + cross-entropy output
- mini-batch training
- Momentum / RMSProp / Adam
- train/validation datasets
- CSV import
- model save/load
- numerical gradient checking
- proper softmax Jacobian-vector products
- architecture presets
- interactive decision-surface plots
- unit tests for backpropagation

## Portfolio note

This is intentionally a **from-scratch neural-network project**. Frameworks such as PyTorch make all of this substantially easier; the value of this implementation is that the weight matrices, activation functions, gradients, parameter updates and neuron responses are directly inspectable rather than delegated to an autograd engine.
