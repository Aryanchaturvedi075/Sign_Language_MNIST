import numpy as np
# #%matplotlib notebook
# %matplotlib inline
import matplotlib.pyplot as plt
# from IPython.core.debugger import set_trace
# import warnings
# warnings.filterwarnings('ignore')
from typing import List
from tqdm import tqdm
from sklearn import datasets
from sklearn.metrics import accuracy_score


class NeuralNetLayer:
    def __init__(self):
        self.gradient = None
        self.parameters = None

    def forward(self, x):                 # given the previous layer neurons, compute the neurons for this layer
        raise NotImplementedError

    def backward(self, gradient):         # given the gradients passed down from the next layer, compute the gradients for this layer's parameters (if applicable) as well as the gradients to pass down to the previous layer
        raise NotImplementedError
    

class LinearLayer(NeuralNetLayer):        #  the only type of layer that has updatable parameters is the linear layer.
    def __init__(self, input_size, output_size):
        super().__init__()
        # self.gradient = None
        # self.parameters = None
        self.ni = input_size
        self.no = output_size
        self.w = np.random.randn(output_size, input_size)         # Matrix of nO x nI dimensions, initialized with random values from the gaussian distribution
        self.b = np.random.randn(output_size)                     # Vector of n0 parameters, initialized with random values from the gaussian distribution
        self.cur_input = None
        self.parameters = [self.w, self.b]                        # parameters[0] stores the matrix of weights for that Linear Layer
                                                                  # parameters[1] stores the vector of biases for all the neurons in the linear Layer
    def forward(self, x):
        self.cur_input = x                                        # update cur_input based on the input neurons
        # return self.w @ x + self.b
        # return np.dot(self.w, x) + self.b                         # x' = Wx + b is passed on forward
        return (self.w[None, :, :] @ x[:, :, None]).squeeze() + self.b

    def backward(self, gradient):
        assert self.cur_input is not None, "Must call forward before backward"
        # dw = gradient.dot(self.cur_input)
        dw = gradient[:, :, None] @ self.cur_input[:, None, :]
        # dw = gradient @ self.cur_input.T                          # compute the gradients for this layer's parameters
        db = gradient                                             # store the changes of gradients to pass down to previous layers too
        self.gradient = [dw, db]
        return gradient.dot(self.w)
    

class ReLULayer(NeuralNetLayer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.gradient = np.where(x > 0, 1.0, 0.0)
        return np.maximum(0, x)

    def backward(self, gradient):
        assert self.gradient is not None, "Must call forward before backward"
        return gradient * self.gradient
    

class SoftmaxOutputLayer(NeuralNetLayer):
    def __init__(self):
        super().__init__()
        self.cur_probs = None

    def forward(self, x):
        exps = np.exp(x)
        probs = exps / np.sum(exps, axis=-1)[:, None]
        self.cur_probs = probs
        return probs

    def backward(self, target):
        assert self.cur_probs is not None, "Must call forward before backward"
        return self.cur_probs - target
    

class MLP:
    def __init__(self, *args: List[NeuralNetLayer]):
        self.layers = args

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, target):
        for layer in self.layers[::-1]:
            target = layer.backward(target)

    def fit(self, optimizer, data_x, data_y, steps):
      losses = []
      labels = np.eye(3)[np.array(data_y)]
      for _ in tqdm(range(steps)):
          predictions = self.forward(data_x)
          loss = -(labels * np.log(predictions)).sum(axis=-1).mean()
          losses.append(loss)
          self.backward(labels)
          optimizer.step()
      # plt.plot(losses)
      # plt.xlabel("Epoch")
      # plt.ylabel("Cross entropy loss")

    def predict(self, x):
      probabilities = self.forward(x)
      return np.argmax(probabilities, axis=-1)


class Optimizer:
    def __init__(self, net: MLP):
        self.net = net

    def step(self):
        for layer in self.net.layers[::-1]:
            if layer.parameters is not None:
                self.update(layer.parameters, layer.gradient)

    def update(self, params, gradient):
        raise NotImplementedError


class GradientDescentOptimizer(Optimizer):
    def __init__(self, net: MLP, lr: float):
        super().__init__(net)
        self.lr = lr

    def update(self, params, gradient):
        for (p, g) in zip(params, gradient):
            p -= self.lr * g.mean(axis=0)


def plot_decision_boundary(mlp: MLP, data_x, data_y):
    x0v = np.linspace(np.min(data_x[:,0]), np.max(data_x[:,0]), 200)
    x1v = np.linspace(np.min(data_x[:,1]), np.max(data_x[:,1]), 200)
    x0,x1 = np.meshgrid(x0v, x1v)
    x_all = np.vstack((x0.ravel(),x1.ravel())).T
    yh_all = np.argmax(mlp.forward(x_all), axis=-1)
    plt.scatter(data_x[:,0], data_x[:,1], c=data_y, marker='o', alpha=1)
    plt.scatter(x_all[:,0], x_all[:,1], c=yh_all, marker='.', alpha=.01)
    plt.ylabel('sepal length')
    plt.xlabel('sepal width')
    plt.title('decision boundary of the MLP')
    plt.show()

if __name__ == '__main__':
    dataset = datasets.load_iris()
    x, y = dataset['data'][:,[1,2]], dataset['target']
    n_features = x.shape[-1]

    HIDDEN_SIZE = 32
    GRADIENT_STEPS = 200

    mlp1 = MLP(
        LinearLayer(n_features, HIDDEN_SIZE),
        ReLULayer(),
        LinearLayer(HIDDEN_SIZE, 3),
        SoftmaxOutputLayer()
    )
    opt1 = GradientDescentOptimizer(mlp1, 1e-1)

    mlp1.fit(opt1, x, y, GRADIENT_STEPS)
    output_mlp1 = mlp1.predict(x[:1000])
    print(output_mlp1)
    # y_pred = np.argmax(output_mlp1, axis=0) #get label
    # y_true = np.argmax(Y_test, axis=1)
    accuracy = accuracy_score(y[:1000], output_mlp1)
    print(accuracy)

    # plot_decision_boundary(mlp1, x, y)

    mlp2 = MLP(
        LinearLayer(n_features, HIDDEN_SIZE),
        ReLULayer(),
        LinearLayer(HIDDEN_SIZE, HIDDEN_SIZE),
        ReLULayer(),
        LinearLayer(HIDDEN_SIZE, 3),
        SoftmaxOutputLayer()
    )
    opt2 = GradientDescentOptimizer(mlp2, 1e-2)
    mlp2.fit(opt2, x, y, GRADIENT_STEPS)
    output_mlp2 = mlp2.predict(x[:1000])
    # y_pred = np.argmax(output_mlp2, axis=0) #get label
    accuracy = accuracy_score(y[:1000], output_mlp2)
    print(output_mlp2)
    print(accuracy)

    # plot_decision_boundary(mlp2, x, y)