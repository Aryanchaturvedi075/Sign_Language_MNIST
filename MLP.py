# %autosave 60
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score

# from keras.preprocessing.image import ImageDataGenerator
# from tensorflow.config import list_physical_devices

# if not list_physical_devices("GPU"):
#     warning = """No GPU was detected. DNNs can be very slow without a GPU.\n
#     Go to Runtime > Change runtime type and select a GPU hardware accelerator."""
#     raise Exception(warning)
# else:
#     print("T4 GPU detected")


class NeuralNetLayer:
    def __init__(self):
        self.gradient = None
        self.parameters = None

    # def forward(self, x):
    #     raise NotImplementedError

    # def backward(self, gradient):
    #     raise NotImplementedError
    

class LinearLayer(NeuralNetLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        # self.ni = input_size
        # self.no = output_size                             # Fourth Key Change
        self.w = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.b = np.random.randn(output_size)
        self.cur_input = None
        self.parameters = [self.w, self.b]

    def forward(self, x):
        self.cur_input = x
        # return self.w @ x + self.b
        # return np.dot(self.w, x) + self.b
        return (self.w[None, :, :] @ x[:, :, None]).squeeze() + self.b

    def backward(self, gradient):
        assert self.cur_input is not None, "Must call forward before backward"
        # dw = gradient.dot(self.cur_input)
        # dw = gradient[:, :, None] @ self.cur_input[:, None, :]
        dw = gradient.T @ self.cur_input
        db = gradient.mean(axis=0)
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
        max_val = np.max(x, axis=1, keepdims=True)                               # Third change
        exps = np.exp(x - max_val)
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
        # labels = np.eye(3)[np.array(data_y)]
        # labels = np.array(data_y)
        labels = data_y
        for _ in tqdm(range(steps)):
            predictions = self.forward(data_x)
            epsilon = 1e-7                                                      # Second Change
            loss = -(labels * np.log(predictions + epsilon)).sum(axis=-1).mean()
            losses.append(loss)
            self.backward(labels)
            optimizer.step()
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Cross entropy loss")
        plt.show()

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

    # def update(self, params, gradient):
    #     raise NotImplementedError


class GradientDescentOptimizer(Optimizer):
    def __init__(self, net: MLP, lr: float):
        super().__init__(net)
        self.lr = lr

    def update(self, params, gradient):
        for (p, g) in zip(params, gradient):
            np.clip(g, -1, 1, out=g)                                            # First Change
            p -= self.lr * g


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

def evaluate_accuracy(y_true , y_pred):
    total_correct_predictions = np.sum(y_true == y_pred)
    overall_accuracy = total_correct_predictions / len(y_true)

    return accuracy

if __name__ == '__main__':
    train_url = 'https://raw.githubusercontent.com/Aryanchaturvedi075/Sign_Language_MNIST/main/sign_mnist_train.csv'
    test_url = 'https://raw.githubusercontent.com/Aryanchaturvedi075/Sign_Language_MNIST/main/sign_mnist_test.csv'
    train_df = pd.read_csv(train_url)
    test_df = pd.read_csv(test_url)

    # For Training Data
    train_label = train_df['label']
    train_set = train_df.drop(['label'], axis=1)

    # For Testing Data
    test_label = test_df['label']
    test_set = test_df.drop(['label'], axis=1)

    X_train = train_set.values.astype(np.float64)
    X_test = test_set.values.astype(np.float64)
    # X_train = trainset.values.reshape(-1,28,28,1).astype(np.float64)
    # X_test = testset.values.reshape(-1,28,28,1).astype(np.float64)

    X_train /= 255
    X_test /= 255
    
    X_train -= np.mean(X_train, axis = 0)
    X_test -= np.mean(X_test, axis = 0)

    # Y_train = train_label.values
    # Y_test = test_label.values
    lb=LabelBinarizer()
    Y_train = lb.fit_transform(train_label)
    Y_test = lb.fit_transform(test_label)

    INPUT_SIZE = 784
    OUTPUT_SIZE = 24
    HIDDEN_SIZE_1 = 32
    HIDDEN_SIZE_2 = 32
    LEARNING_RATE_1 = 1e-2
    LEARNING_RATE_2 = 1e-2
    GRADIENT_STEPS = 20

    mlp1 = MLP(
        LinearLayer(INPUT_SIZE, HIDDEN_SIZE_1),
        ReLULayer(),
        LinearLayer(HIDDEN_SIZE_1, OUTPUT_SIZE),
        SoftmaxOutputLayer()
    )
    opt1 = GradientDescentOptimizer(mlp1, LEARNING_RATE_1)

    mlp2 = MLP(
        LinearLayer(INPUT_SIZE, HIDDEN_SIZE_1),
        ReLULayer(),
        LinearLayer(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
        ReLULayer(),
        LinearLayer(HIDDEN_SIZE_2, OUTPUT_SIZE),
        SoftmaxOutputLayer()
    )
    opt2 = GradientDescentOptimizer(mlp2, LEARNING_RATE_2)

    mlp1.fit(opt1, X_train, Y_train, GRADIENT_STEPS)
    output_mlp1 = mlp1.predict(X_test)
    # y_pred = np.argmax(output_mlp1, axis=0) #get label
    # y_true = np.argmax(Y_test, axis=1)
    accuracy = accuracy_score(Y_test, output_mlp1)
    print(accuracy * 100)
    # plot_decision_boundary(mlp1, X_train, Y_train)

    # mlp2.fit(opt2, X_train, Y_train, GRADIENT_STEPS)
    # output_mlp2 = mlp2.predict(X_test)
    # y_pred = np.argmax(output_mlp2, axis=0) #get label
    # accuracy = accuracy_score(Y_test, output_mlp2)
    # print(accuracy)
    # plot_decision_boundary(mlp2, X_train, Y_train)