import numpy as np


# sigmoid: sigmoid function for matrices
# Parameters
#   X: np.ndarray with shape [M, N]
# Output
#   Y: np.ndarray with shape [M, N] s.t. Y[i,j] = sigmoid(X[i,j])
def sigmoid(X):
    Y = X
    
    # TODO: IMPLEMENT ME
    pass


# dsigmoid: the derivative of sigmoid function for matrices
# Parameters
#   X: np.ndarray with shape [M, N]
# Output
#   Y: np.ndarray with shape [M, N] s.t.
#     Y[i,j] is the derivative of sigmoid at X[i,j]
# C.f. the derivative of sigmoid is found at
#   https://en.wikipedia.org/wiki/Activation_function
def dsigmoid(X):
    # TODO: IMPLEMENT ME
    pass


# softmax: softmax function for a bunch of data points
# Parameters
#   X: np.ndarray with shape [# of data points, # of features]
# Output
#   Y: np.ndarray with shape [# of data points, probabilies] s.t.
#     Y[i,j] = softmax^j(X[i])
def softmax(X):
    # TODO: IMPLEMENT ME
    pass


class FeedforwardNeuralNetwork:
    # Parameters
    #   n_neurons1: the # of neurons at the 1st hidden layer
    #   n_neurons2: the # of neurons at the 2nd hidden layer
    #   n_epochs:   the # of iterations to run gradient descent
    #   lr = 0.001: learning rate
    #   print_loss: whether loss values during training are printed
    def __init__(self, n_neurons1, n_neurons2, n_epochs, lr,
                 print_loss=False):
        self.n_neurons1 = n_neurons1
        self.n_neurons2 = n_neurons2
        self.n_epochs = n_epochs
        self.lr = lr
        self.print_loss = print_loss

    # fit: train this model on training inputs X and outputs Y
    # Parameters
    #   X: training inputs
    #     np.ndarray (shape: [# of data points, # of features])
    #   Y: training outputs
    #     np.ndarray (shape: [# of data points])
    def fit(self, X, Y):
        n_features = X.shape[1]
        n_classes = np.max(Y) + 1
        n_data = len(Y)
        self.W1 = np.random.randn(n_features, self.n_neurons1)
        self.B1 = np.random.rand(1, self.n_neurons1)
        self.W2 = np.random.randn(self.n_neurons1, self.n_neurons2)
        self.B2 = np.random.rand(1, self.n_neurons2)
        self.W3 = np.random.randn(self.n_neurons2, n_classes)
        self.B3 = np.random.rand(1, n_classes)

        for i in range(self.n_epochs):
            A0 = X
            # TODO: IMPLEMENT Z1, Z2, Z3
            # Zi is the result of applying linear function with
            #   weights self.Wi and biases self.Bi to Ai-1
            # Ai is the result of applying an activation function to Zi
            #  Activation functions for A1 and A2: sigmoid
            #  Activation functions for A3: softmax
            Z1 = np.zeros((n_data, self.n_neurons1))  # dummy
            A1 = sigmoid(Z1)  # output from the 1st hidden layer
            Z2 = np.zeros((n_data, self.n_neurons2))  # dummy
            A2 = sigmoid(Z2)  # outoput from the 2nd hidden layer
            Z3 = np.zeros((n_data, self.n_classes))   # dummy
            A3 = softmax(Z3)  # outoupt from the output layer

            # Convert class labels to one-hot vectors
            expected = np.zeros((n_data, n_classes))
            expected[np.arange(n_data), Y] = 1

            if self.print_loss:
                loss = - np.sum(np.log(A3) * expected) / n_data
                print('loss', loss, '@', i, 'epoch')

            # Update parameters by gradient descent
            grad_Z3 = A3 - expected
            grad_W3, grad_B3 = self.grad_params(grad_Z3, A2)
            self.W3 -= self.lr * grad_W3
            self.B3 -= self.lr * grad_B3

            grad_Z2 = np.dot(grad_Z3, self.W3.T) * dsigmoid(Z2)
            grad_W2, grad_B2 = self.grad_params(grad_Z2, A1)
            self.W2 -= self.lr * grad_W2
            self.B2 -= self.lr * grad_B2

            grad_Z1 = np.dot(grad_Z2, self.W2.T) * dsigmoid(Z1)
            grad_W1, grad_B1 = self.grad_params(grad_Z1, X)
            self.W1 -= self.lr * grad_W1
            self.B1 -= self.lr * grad_B1

    # grad_params: calculates gradients of weights W and biases B of a layer
    # Parameters
    #   A: inputs to the layer (shape: [# of data points, # of features])
    #   grad_Z: the derivative of the cost function at Z = A * W + B
    #     (`*` is the matrix multiplicaiton; shape: [# of data points, # of features])
    # Output
    #   grad_W: gradients of W
    #   grad_B: gradients of B
    def grad_params(self, grad_Z, A):
        assert grad_Z.shape[0] == A.shape[0]
        n_data = grad_Z.shape[0]
        n_features_Z = grad_Z.shape[1]
        n_features_A = A.shape[1]

        grad_W = np.sum(np.dot(A.T, grad_Z), axis=0) / n_data
        grad_B = np.sum(grad_Z, axis=0, keepdims=True) / n_data

        return grad_W, grad_B

    # predict: classify given data points
    # Parameters
    #   X: inputs to the classifier
    #     np.ndarray (shape: [# of data points, # of features])
    # Output
    #   Y: classificaiton results
    #     np.ndarray (shape: [# of data points]) s.t.
    #     Y[i] is the label predicted for data point X[i]
    def predict(self, X):
        # TODO: IMPLEMENT ME
        # Hint
        #   self.W1, self.W2: weight parameters at the 1st and 2nd layer, resp.
        #   self.B1, self.B2: bias parameters at the 1st and 2nd layer, resp.
        #   Activation functions at hidden layers are sigmoid
        #   The activation function at the output layer is softmax
        #
        # Tips
        #   `np.argmax` may be useful.
        #   Let A be np.ndarray with shape [M, N].  Then, `np.argmax(A, axis=1)`
        #   returns the np.ndarray B with shape [M] s.t. B[i] is the index of
        #   the maximum value among A[i]
        pass


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    np.random.seed(0)

    iris_dataset = load_iris()
    X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset.data,
                                                        iris_dataset.target,
                                                        random_state=0)
    # If print_loss is set to True, the loss values will be shown
    print_loss = False
    n_neurons1 = 100  # the # of neurons at the 1st hidden layer
    n_neurons2 = 100  # the # of neurons at the 2nd hidden layer
    n_epochs = 1000   # the # of iterations to run gradient descent
    lr = 0.001        # learning rate
    fnn = FeedforwardNeuralNetwork(n_neurons1, n_neurons2, n_epochs, lr,
                                   print_loss=print_loss)
    fnn.fit(X_train, Y_train)

    X_test_predict = fnn.predict(X_test)
    accuracy = np.sum(Y_test == X_test_predict) / Y_test.shape[0]
    assert accuracy > 0.7
    print('acc:', accuracy)
