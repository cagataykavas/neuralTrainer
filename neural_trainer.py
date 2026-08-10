import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk


# ============================================================
# Activation functions and derivatives
# ============================================================

def relu(x):
    return np.maximum(0, x)


def drelu(x):
    return np.where(x > 0, 1.0, 0.0)


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(x):
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x):
    return np.tanh(x)


def dtanh(x):
    return 1.0 - np.tanh(x) ** 2


def softmax(x):
    ex = np.exp(x - np.max(x, axis=0, keepdims=True))
    return ex / np.sum(ex, axis=0, keepdims=True)


def dsoftmax(x):
    """Element-wise approximation.

    A full softmax derivative is a Jacobian. This helper is preserved for
    educational experimentation, but the current network uses an identity
    output layer and therefore does not depend on this approximation there.
    """
    s = softmax(x)
    return s * (1.0 - s)


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def dsoftplus(x):
    return sigmoid(x)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def dleaky_relu(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)


def get_activation_funcs(name):
    name = name.lower()
    if name == "relu":
        return relu, drelu
    if name == "sigmoid":
        return sigmoid, dsigmoid
    if name == "tanh":
        return tanh, dtanh
    if name == "softmax":
        return softmax, dsoftmax
    if name == "softplus":
        return softplus, dsoftplus
    if name in {"leakyrelu", "leaky_relu"}:
        return leaky_relu, dleaky_relu
    return relu, drelu


# ============================================================
# Error functions and derivatives
# ============================================================

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mse_derivative(y_true, y_pred):
    return 2.0 * (y_pred - y_true) / y_true.size


def me(y_true, y_pred):
    return np.mean(y_true - y_pred)


def me_derivative(y_true, y_pred):
    # d/dy_pred mean(y_true - y_pred) = -1/N
    return -np.ones_like(y_pred) / y_true.size


def log_likelihood(y_true, y_pred):
    clipped = np.clip(y_pred, 1e-9, None)
    return -np.mean(np.sum(y_true * np.log(clipped), axis=0))


def log_likelihood_derivative(y_true, y_pred):
    clipped = np.clip(y_pred, 1e-9, None)
    return -(y_true / clipped) / y_true.size


def manhattan_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def manhattan_error_derivative(y_true, y_pred):
    return np.sign(y_pred - y_true) / y_true.size


def log_error(y_true, y_pred):
    return np.mean(np.log1p(np.abs(y_true - y_pred)))


def log_error_derivative(y_true, y_pred):
    diff = y_pred - y_true
    return (np.sign(diff) / (np.abs(diff) + 1.0)) / y_true.size


def get_error_funcs(name):
    name = name.lower()
    if name == "mse":
        return mse, mse_derivative
    if name == "me":
        return me, me_derivative
    if name in {"log_likelihood", "loglikelihood"}:
        return log_likelihood, log_likelihood_derivative
    if name in {"manhattan", "manhattan_error", "l1"}:
        return manhattan_error, manhattan_error_derivative
    if name in {"log_error", "logarithmic"}:
        return log_error, log_error_derivative
    return mse, mse_derivative


# ============================================================
# Neural network with explicit backpropagation
# ============================================================

class NeuralNetwork:
    def __init__(
        self,
        input_size,
        hidden_layers,
        output_size,
        error_name="mse",
        learning_rate=0.00005,
        manual_params=None,
        activations=None,
        seed=None,
    ):
        self.input_size = input_size
        self.hidden_layers = list(hidden_layers)
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.error_func, self.error_deriv = get_error_funcs(error_name)
        self.error_name = error_name
        self.activations = activations or ["leaky_relu"] * len(hidden_layers)

        if len(self.activations) != len(self.hidden_layers):
            raise ValueError("Provide exactly one activation per hidden layer.")

        rng = np.random.default_rng(seed)
        layer_sizes = [input_size] + self.hidden_layers + [output_size]
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            if manual_params and i < len(manual_params):
                w, b = manual_params[i]
                w = np.asarray(w, dtype=float)
                b = np.asarray(b, dtype=float)
            else:
                w = rng.uniform(-5, 5, (layer_sizes[i + 1], layer_sizes[i]))
                b = rng.uniform(-5, 5, (layer_sizes[i + 1], 1))

            expected_w = (layer_sizes[i + 1], layer_sizes[i])
            expected_b = (layer_sizes[i + 1], 1)
            if w.shape != expected_w:
                raise ValueError(f"Layer {i} weight shape {w.shape}; expected {expected_w}.")
            if b.shape != expected_b:
                raise ValueError(f"Layer {i} bias shape {b.shape}; expected {expected_b}.")

            self.weights.append(w.copy())
            self.biases.append(b.copy())

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != self.input_size:
            raise ValueError(f"Expected {self.input_size} input rows, got {X.shape[0]}.")

        activations = [X]
        pre_activations = []
        A = X

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            Z = w @ A + b
            pre_activations.append(Z)

            if i == len(self.weights) - 1:
                A = Z  # identity output layer
            else:
                act_func, _ = get_activation_funcs(self.activations[i])
                A = act_func(Z)

            activations.append(A)

        return activations, pre_activations

    def predict(self, X):
        return self.forward(X)[0][-1]

    def backward(self, activations, pre_activations, Y):
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        A_final = activations[-1]
        d_loss = self.error_deriv(Y, A_final)
        dZ = d_loss

        grads_w[-1] = dZ @ activations[-2].T
        grads_b[-1] = np.sum(dZ, axis=1, keepdims=True)

        for layer in range(len(self.weights) - 2, -1, -1):
            dA = self.weights[layer + 1].T @ dZ
            _, activation_derivative = get_activation_funcs(self.activations[layer])
            dZ = dA * activation_derivative(pre_activations[layer])
            grads_w[layer] = dZ @ activations[layer].T
            grads_b[layer] = np.sum(dZ, axis=1, keepdims=True)

        return grads_w, grads_b

    def update_params(self, grads_w, grads_b, clip_value=10.0):
        for i in range(len(self.weights)):
            grad_w = np.clip(grads_w[i], -clip_value, clip_value)
            grad_b = np.clip(grads_b[i], -clip_value, clip_value)
            self.weights[i] -= self.learning_rate * grad_w
            self.biases[i] -= self.learning_rate * grad_b

    def train(self, X, Y, iterations, callback=None):
        losses = []
        for i in range(iterations):
            activations, pre_activations = self.forward(X)
            loss = float(self.error_func(Y, activations[-1]))
            grads_w, grads_b = self.backward(activations, pre_activations, Y)
            self.update_params(grads_w, grads_b)
            losses.append(loss)

            if callback is not None:
                callback(i, losses, self)

        return losses


# ============================================================
# Tkinter GUI and visualizations
# ============================================================

class NeuralNetGUI:
    def __init__(self, master):
        self.master = master
        master.title("Neural Network Trainer")

        self.frame_main = ttk.Frame(master, padding="10")
        self.frame_main.grid(row=0, column=0, sticky="N")

        ttk.Label(self.frame_main, text="Input Size:").grid(row=0, column=0, sticky="W")
        self.spin_input_size = ttk.Spinbox(
            self.frame_main, from_=1, to=20, width=5, command=self.update_dynamic_inputs
        )
        self.spin_input_size.set(3)
        self.spin_input_size.grid(row=0, column=1, sticky="W")

        ttk.Label(self.frame_main, text="Output Size:").grid(row=1, column=0, sticky="W")
        self.spin_output_size = ttk.Spinbox(
            self.frame_main, from_=1, to=10, width=5, command=self.output_size_changed
        )
        self.spin_output_size.set(1)
        self.spin_output_size.grid(row=1, column=1, sticky="W")

        ttk.Label(self.frame_main, text="Number of Hidden Layers:").grid(
            row=2, column=0, sticky="W"
        )
        self.spin_hidden_layers = ttk.Spinbox(
            self.frame_main, from_=0, to=10, width=5, command=self.update_hidden_layer_config
        )
        self.spin_hidden_layers.set(2)
        self.spin_hidden_layers.grid(row=2, column=1, sticky="W")

        ttk.Label(self.frame_main, text="Learning Rate:").grid(row=3, column=0, sticky="W")
        self.learning_rate = ttk.Entry(self.frame_main, width=10)
        self.learning_rate.insert(0, "0.00005")
        self.learning_rate.grid(row=3, column=1, sticky="W")

        ttk.Label(self.frame_main, text="Loss Function:").grid(row=4, column=0, sticky="W")
        self.combo_error = ttk.Combobox(
            self.frame_main,
            values=["mse", "me", "log_likelihood", "manhattan", "log_error"],
            width=14,
            state="readonly",
        )
        self.combo_error.current(0)
        self.combo_error.grid(row=4, column=1, sticky="W")

        ttk.Label(self.frame_main, text="Iterations:").grid(row=5, column=0, sticky="W")
        self.spin_iterations = ttk.Spinbox(self.frame_main, from_=1, to=10000, width=7)
        self.spin_iterations.set(500)
        self.spin_iterations.grid(row=5, column=1, sticky="W")

        self.rand_init = tk.BooleanVar(value=True)
        self.chk_rand = ttk.Checkbutton(
            self.frame_main,
            text="Random Initialization",
            variable=self.rand_init,
            command=self.toggle_manual_params,
        )
        self.chk_rand.grid(row=6, column=0, columnspan=2, sticky="W")

        vis_frame = ttk.LabelFrame(self.frame_main, text="Visualization Options")
        vis_frame.grid(row=7, column=0, columnspan=2, pady=5, sticky="W")

        self.show_vis = tk.BooleanVar(value=True)
        ttk.Checkbutton(vis_frame, text="Live Visualization", variable=self.show_vis).grid(
            row=0, column=0, sticky="W"
        )

        ttk.Label(vis_frame, text="Update Every:").grid(
            row=0, column=1, padx=(10, 0), sticky="W"
        )
        self.vis_interval = ttk.Spinbox(vis_frame, from_=1, to=100, width=5)
        self.vis_interval.set(1)
        self.vis_interval.grid(row=0, column=2, padx=2, sticky="W")
        ttk.Label(vis_frame, text="iterations").grid(row=0, column=3, sticky="W")

        self.frame_train = ttk.LabelFrame(self.frame_main, text="Training Data")
        self.frame_train.grid(row=8, column=0, columnspan=2, pady=5, sticky="W")
        ttk.Label(self.frame_train, text="Input/s:").grid(row=0, column=0, sticky="W")
        ttk.Label(self.frame_train, text="Target/s:").grid(row=1, column=0, sticky="W")
        self.dynamic_inputs = []
        self.dynamic_targets = []
        self.update_dynamic_inputs()

        self.btn_train = ttk.Button(
            self.frame_main, text="Train & Visualize", command=self.train_network
        )
        self.btn_train.grid(row=9, column=0, columnspan=2, pady=10)

        ttk.Separator(self.frame_main, orient="horizontal").grid(
            row=10, column=0, columnspan=2, sticky="ew", pady=5
        )

        cont_frame = ttk.LabelFrame(self.frame_main, text="Network Response Visualization")
        cont_frame.grid(row=11, column=0, columnspan=2, pady=5, sticky="W")

        ttk.Label(cont_frame, text="Input Range:").grid(row=0, column=0, sticky="W")
        ttk.Label(cont_frame, text="From:").grid(row=0, column=1, sticky="E")
        self.range_min = ttk.Entry(cont_frame, width=5)
        self.range_min.insert(0, "-10")
        self.range_min.grid(row=0, column=2, padx=2)

        ttk.Label(cont_frame, text="To:").grid(row=0, column=3, sticky="E")
        self.range_max = ttk.Entry(cont_frame, width=5)
        self.range_max.insert(0, "10")
        self.range_max.grid(row=0, column=4, padx=2)

        ttk.Label(cont_frame, text="Points:").grid(row=0, column=5, sticky="E")
        self.range_points = ttk.Entry(cont_frame, width=5)
        self.range_points.insert(0, "400")
        self.range_points.grid(row=0, column=6, padx=2)

        self.real_time_viz = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            cont_frame,
            text="Real-time Response Visualization",
            variable=self.real_time_viz,
        ).grid(row=1, column=0, columnspan=7, pady=5)

        self.frame_activations = ttk.LabelFrame(
            master, text="Neural Network Structure & Configuration"
        )
        self.frame_activations.grid(row=1, column=0, padx=10, pady=10, sticky="EW")

        self.frame_activation_controls = ttk.Frame(self.frame_activations)
        self.frame_activation_controls.grid(row=0, column=0, sticky="W")

        self.frame_manual = ttk.LabelFrame(self.frame_activations, text="Manual Weights/Biases")
        self.frame_manual.grid(row=1, column=0, pady=10, sticky="EW")
        self.frame_manual.grid_remove()

        self.hidden_layer_entries = {}
        self.activation_dropdowns = {}
        self.manual_entries = {}
        self.update_activation_dropdowns()

        self.last_trained_network = None
        self.fig_response = None
        self.axs_response = None
        self.fig_network = None
        self.ax_network = None

    def update_dynamic_inputs(self):
        for widget in self.dynamic_inputs + self.dynamic_targets:
            widget.destroy()

        self.dynamic_inputs = []
        self.dynamic_targets = []

        input_size = int(self.spin_input_size.get())
        output_size = int(self.spin_output_size.get())

        for i in range(input_size):
            entry = ttk.Entry(self.frame_train, width=5)
            entry.grid(row=0, column=i + 1, padx=2)
            entry.insert(0, "0")
            self.dynamic_inputs.append(entry)

        for i in range(output_size):
            entry = ttk.Entry(self.frame_train, width=5)
            entry.grid(row=1, column=i + 1, padx=2)
            entry.insert(0, "0")
            self.dynamic_targets.append(entry)

    def toggle_manual_params(self):
        if self.rand_init.get():
            self.frame_manual.grid_remove()
        else:
            self.frame_manual.grid()
            self.populate_manual_params()

    def populate_manual_params(self):
        for child in self.frame_manual.winfo_children():
            child.destroy()

        input_size = int(self.spin_input_size.get())
        num_hidden = int(self.spin_hidden_layers.get())
        hidden_sizes = [int(self.hidden_layer_entries[i].get()) for i in range(num_hidden)]
        output_size = int(self.spin_output_size.get())
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.manual_entries = {}

        for layer in range(len(layer_sizes) - 1):
            layer_frame = ttk.LabelFrame(
                self.frame_manual,
                text=f"Layer {layer} → {layer + 1} ({layer_sizes[layer]} → {layer_sizes[layer + 1]})",
            )
            layer_frame.grid(row=layer, column=0, padx=5, pady=5, sticky="W")

            weight_frame = ttk.Frame(layer_frame)
            weight_frame.grid(row=0, column=0, padx=5, pady=5)
            ttk.Label(weight_frame, text="Weights (Ω):").grid(row=0, column=0, sticky="W")

            matrix_frame = ttk.Frame(weight_frame)
            matrix_frame.grid(row=1, column=0, padx=5, pady=5)

            weight_entries = []
            for row in range(layer_sizes[layer + 1]):
                row_entries = []
                for col in range(layer_sizes[layer]):
                    entry = ttk.Entry(matrix_frame, width=5)
                    entry.grid(row=row, column=col, padx=1, pady=1)
                    entry.insert(0, "0")
                    row_entries.append(entry)
                weight_entries.append(row_entries)

            bias_frame = ttk.Frame(layer_frame)
            bias_frame.grid(row=0, column=1, padx=5, pady=5)
            ttk.Label(bias_frame, text="Biases (θ):").grid(row=0, column=0, sticky="W")

            biases_frame = ttk.Frame(bias_frame)
            biases_frame.grid(row=1, column=0, padx=5, pady=5)
            bias_entries = []
            for row in range(layer_sizes[layer + 1]):
                entry = ttk.Entry(biases_frame, width=5)
                entry.grid(row=row, column=0, padx=1, pady=1)
                entry.insert(0, "0")
                bias_entries.append(entry)

            self.manual_entries[layer] = (weight_entries, bias_entries)

    def update_hidden_layer_config(self):
        self.update_activation_dropdowns()
        if not self.rand_init.get():
            self.populate_manual_params()

    def update_activation_dropdowns(self):
        for child in self.frame_activation_controls.winfo_children():
            child.destroy()

        self.hidden_layer_entries = {}
        self.activation_dropdowns = {}
        activation_options = ["relu", "sigmoid", "tanh", "softmax", "softplus", "leaky_relu"]
        num_hidden = int(self.spin_hidden_layers.get())

        for layer in range(num_hidden):
            frame_col = ttk.Frame(self.frame_activation_controls)
            frame_col.grid(row=0, column=layer, padx=5, pady=5)

            ttk.Label(frame_col, text=f"Hidden Layer {layer + 1} Neurons:").grid(row=0, column=0)
            entry = ttk.Entry(frame_col, width=5)
            entry.insert(0, "4")
            entry.grid(row=1, column=0)
            entry.bind("<FocusOut>", lambda event, idx=layer: self.neuron_count_changed(idx))
            self.hidden_layer_entries[layer] = entry

            ttk.Label(frame_col, text="Activation:").grid(row=2, column=0)
            dropdown = ttk.Combobox(
                frame_col, values=activation_options, width=10, state="readonly"
            )
            dropdown.set("leaky_relu")
            dropdown.grid(row=3, column=0)
            self.activation_dropdowns[layer] = dropdown

    def neuron_count_changed(self, _layer_idx):
        if not self.rand_init.get():
            self.populate_manual_params()

    def plot_network_structure(self, nn, X=None, update=False):
        layers = []

        if X is not None:
            inputs = X.flatten()
            input_labels = [f"{value:.2f}" for value in inputs]
            num_input = len(inputs)
        else:
            num_input = nn.input_size
            input_labels = [f"I{i + 1}" for i in range(num_input)]

        layers.append((num_input, 0, input_labels, "Input"))
        for i, hidden_size in enumerate(nn.hidden_layers):
            layers.append((hidden_size, i + 1, None, f"Hidden Layer {i + 1}"))
        layers.append((nn.output_size, len(nn.hidden_layers) + 1, None, "Output Layer"))

        positions = []
        for num_nodes, x_coord, labels, title in layers:
            if num_nodes > 1:
                y_values = np.linspace(0, num_nodes - 1, num_nodes) - (num_nodes - 1) / 2
            else:
                y_values = np.array([0.0])
            positions.append((x_coord, y_values, labels, title))

        computed_activations = nn.forward(X)[0] if X is not None else None

        if update and self.fig_network is not None and self.ax_network is not None:
            fig, ax = self.fig_network, self.ax_network
            ax.clear()
        else:
            fig, ax = plt.subplots(figsize=(9, 6))
            self.fig_network, self.ax_network = fig, ax

        node_radius = 0.2

        for layer_index, (x_coord, y_values, labels, title) in enumerate(positions):
            y_bottom = np.min(y_values) - node_radius - 0.3
            ax.text(x_coord, y_bottom, title, ha="center", va="top", fontsize=10, fontweight="bold")

            for node_index, y_coord in enumerate(y_values):
                neuron = plt.Circle((x_coord, y_coord), node_radius, fill=True, color="skyblue", ec="black")
                ax.add_patch(neuron)

                if computed_activations is not None:
                    value = computed_activations[layer_index][node_index, 0]
                    ax.text(x_coord, y_coord, f"{value:.2f}", ha="center", va="center", fontsize=8)
                elif layer_index == 0 and labels is not None:
                    ax.text(x_coord, y_coord, labels[node_index], ha="center", va="center", fontsize=8)

                if layer_index > 0:
                    bias = nn.biases[layer_index - 1][node_index, 0]
                    bias_radius = 0.1
                    bias_x = x_coord - node_radius - bias_radius - 0.05
                    bias_node = plt.Circle((bias_x, y_coord), bias_radius, fill=True, color="blue", ec="black")
                    ax.add_patch(bias_node)
                    ax.plot([bias_x + bias_radius, x_coord - node_radius], [y_coord, y_coord], "k-", lw=0.5)
                    ax.text(bias_x, y_coord - 0.2, f"{bias:.2f}", ha="center", va="top", fontsize=6, color="blue")

        for layer in range(len(positions) - 1):
            x1, y_values_1, _, _ = positions[layer]
            x2, y_values_2, _, _ = positions[layer + 1]
            for i, y1 in enumerate(y_values_1):
                for j, y2 in enumerate(y_values_2):
                    ax.plot([x1, x2], [y1, y2], "k-", lw=0.4, alpha=0.55)
                    weight = nn.weights[layer][j, i]
                    wx = x1 + (x2 - x1) * 0.3
                    wy = y1 + (y2 - y1) * 0.3
                    ax.text(wx, wy, f"{weight:.2f}", fontsize=6, color="red")

        ax.set_title(f"Neural Network Structure ({nn.error_name} loss)")
        ax.axis("equal")
        ax.axis("off")
        fig.canvas.draw_idle()
        plt.pause(0.01)
        return fig, ax

    def _response_data(self, nn):
        x_min = float(self.range_min.get())
        x_max = float(self.range_max.get())
        num_points = int(self.range_points.get())
        x_range = np.linspace(x_min, x_max, num_points)

        inputs = np.zeros((nn.input_size, num_points))
        inputs[0, :] = x_range

        layer_outputs = [inputs]
        current_output = inputs

        for i in range(len(nn.weights)):
            z = nn.weights[i] @ current_output + nn.biases[i]
            if i < len(nn.weights) - 1:
                activation, _ = get_activation_funcs(nn.activations[i])
                current_output = activation(z)
            else:
                current_output = z
            layer_outputs.append(current_output)

        return x_range, layer_outputs

    def create_network_response_plot(self):
        if self.fig_response is not None:
            plt.close(self.fig_response)

        nn = self.last_trained_network
        if nn is None:
            return None, None

        x_range, layer_outputs = self._response_data(nn)
        fig, axes = plt.subplots(len(layer_outputs), 1, figsize=(8, 2.4 * len(layer_outputs)))
        axes = np.atleast_1d(axes)

        for i, layer_output in enumerate(layer_outputs):
            ax = axes[i]
            for j in range(layer_output.shape[0]):
                ax.plot(x_range, layer_output[j, :], label=f"Neuron {j + 1}")

            if i == 0:
                title = "Input Layer"
            elif i == len(layer_outputs) - 1:
                title = "Output Layer"
            else:
                title = f"Hidden Layer {i} ({nn.activations[i - 1]})"

            ax.set_title(title)
            ax.set_xlabel("First Input Value")
            ax.set_ylabel("Activation")
            ax.grid(True)
            ax.legend(loc="best")

        plt.tight_layout()
        return fig, axes

    def update_network_response_plot(self, nn, epoch=None):
        if not self.real_time_viz.get() and epoch is None:
            return

        self.last_trained_network = nn
        x_range, layer_outputs = self._response_data(nn)

        if self.fig_response is None or self.axs_response is None:
            self.fig_response, self.axs_response = self.create_network_response_plot()
            if self.fig_response is None:
                return

        axes = np.atleast_1d(self.axs_response)
        for ax in axes:
            ax.clear()

        for i, layer_output in enumerate(layer_outputs):
            ax = axes[i]
            for j in range(layer_output.shape[0]):
                ax.plot(x_range, layer_output[j, :], label=f"Neuron {j + 1}")

            if i == 0:
                title = "Input Layer"
            elif i == len(layer_outputs) - 1:
                title = "Output Layer"
            else:
                title = f"Hidden Layer {i} ({nn.activations[i - 1]})"

            if epoch is not None:
                title += f" - Epoch {epoch}"

            ax.set_title(title)
            ax.set_xlabel("First Input Value")
            ax.set_ylabel("Activation")
            ax.grid(True)
            ax.legend(loc="best")

        plt.tight_layout()
        self.fig_response.canvas.draw_idle()
        plt.pause(0.01)

    def train_network(self):
        input_size = int(self.spin_input_size.get())
        output_size = int(self.spin_output_size.get())
        num_hidden = int(self.spin_hidden_layers.get())
        hidden_layers = [int(self.hidden_layer_entries[i].get()) for i in range(num_hidden)]
        activations = [self.activation_dropdowns[i].get() for i in range(num_hidden)]
        error_name = self.combo_error.get().strip()
        iterations = int(self.spin_iterations.get())
        learning_rate = float(self.learning_rate.get())

        X = np.array([float(entry.get()) for entry in self.dynamic_inputs], dtype=float).reshape(input_size, 1)
        Y = np.array([float(entry.get()) for entry in self.dynamic_targets], dtype=float).reshape(output_size, 1)

        manual_params = None
        if not self.rand_init.get():
            manual_params = []
            for layer in range(len([input_size] + hidden_layers)):
                weight_entries, bias_entries = self.manual_entries[layer]
                weights = [[float(entry.get()) for entry in row] for row in weight_entries]
                biases = [[float(entry.get())] for entry in bias_entries]
                manual_params.append((weights, biases))

        nn = NeuralNetwork(
            input_size=input_size,
            hidden_layers=hidden_layers,
            output_size=output_size,
            error_name=error_name,
            learning_rate=learning_rate,
            manual_params=manual_params,
            activations=activations,
        )

        show_visualization = self.show_vis.get()
        update_interval = max(1, int(self.vis_interval.get()))
        self.last_trained_network = nn

        if self.real_time_viz.get():
            self.fig_response, self.axs_response = self.create_network_response_plot()

        def training_callback(iteration, losses, nn_state):
            if iteration % update_interval != 0:
                return

            if show_visualization:
                _, ax = self.plot_network_structure(nn_state, X, update=True)
                ax.text(
                    0.95,
                    0.95,
                    f"Epoch: {iteration}",
                    transform=ax.transAxes,
                    fontsize=12,
                    ha="right",
                    color="blue",
                )

            if self.real_time_viz.get():
                self.update_network_response_plot(nn_state, epoch=iteration)

        losses = nn.train(X, Y, iterations, callback=training_callback)

        fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
        ax_loss.plot(np.arange(iterations), losses)
        ax_loss.set_title(f"Loss ({error_name}) over iterations")
        ax_loss.set_xlabel("Iteration")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True)
        plt.tight_layout()

        self.plot_network_structure(nn, X)

        if self.fig_response is not None:
            plt.close(self.fig_response)
        self.fig_response = None
        self.axs_response = None
        self.update_network_response_plot(nn, epoch="Final")

        plt.show()

    def output_size_changed(self):
        self.update_dynamic_inputs()
        if not self.rand_init.get():
            self.populate_manual_params()


def main():
    root = tk.Tk()
    NeuralNetGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
