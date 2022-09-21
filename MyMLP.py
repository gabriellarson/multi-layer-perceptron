import numpy as np

class Normalization:
    def __init__(self,):
        self.mean = np.zeros([1,64]) # means of training features
        self.std = np.zeros([1,64]) # standard deviation of training features

    def fit(self,x):
        # compute the statistics of training samples
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

    def normalize(self,x):
        # normalize the given samples to have zero mean and unit variance (add 1e-15 to std to avoid numeric issue)
        return (x-self.mean)/(self.std + (1e-15))

def process_label(label):
    # convert the labels into one-hot vector for training
    one_hot = np.zeros([len(label),10])
    for i in range(len(label)):
        one_hot[i][label[i]] = 1
    return one_hot

def tanh(x):
    #the hyperbolic tangent activation function for hidden layer
    x = np.clip(x,a_min=-100,a_max=100) # for stablility, do not remove this line

    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def softmax(x):
    #the softmax activation function for output layer
    return np.exp(x-np.max(x, axis=-1, keepdims=True)) / np.sum(np.exp(x-np.max(x, axis=-1, keepdims=True)), axis=-1, keepdims=True)

class MLP:
    def __init__(self,num_hid):
        # initialize the weights
        self.weight_1 = np.random.random([64,num_hid])
        self.bias_1 = np.random.random([1,num_hid])
        self.weight_2 = np.random.random([num_hid,10])
        self.bias_2 = np.random.random([1,10])

    def fit(self,train_x,train_y, valid_x, valid_y):
        # learning rate
        lr = 5e-3
        # counter for recording the number of epochs without improvement
        count = 0
        best_valid_acc = 0

        """
        Stop the training if there is no improvment over the best validation accuracy for more than 50 iterations
        """
        while count<=50:
            # training with all samples (full-batch gradient descents)
            # the forward pass (from inputs to predictions)
            hidden_input = train_x.dot(self.weight_1) + self.bias_1
            hidden_output = tanh(hidden_input)

            output_input = hidden_output.dot(self.weight_2) + self.bias_2
            y = softmax(output_input)

            # backpropagation
            # compute the gradients w.r.t. different parameters
            loss = y - train_y
            activation_gradient = y*(1 - y)

            grad_output = loss * activation_gradient
            g_v = hidden_output.T.dot(grad_output)
            g_v_0 = np.sum(grad_output, axis=0, keepdims=True)

            grad_hidden = grad_output.dot(self.weight_2.T) * (1 - np.power(hidden_output, 2))
            g_w = train_x.T.dot(grad_hidden)
            g_w_0 = np.sum(grad_hidden, axis=0, keepdims=True)

            # update the parameters based on sum of gradients for all training samples
            self.weight_1 -= lr*g_w
            self.bias_1 -= lr*g_w_0
            self.weight_2 -= lr*g_v
            self.bias_2 -= lr*g_v_0

            # evaluate on validation data
            predictions = self.predict(valid_x)
            valid_acc = np.count_nonzero(predictions.reshape(-1)==valid_y.reshape(-1))/len(valid_x)

            # compare the current validation accuracy with the best one
            if valid_acc>best_valid_acc:
                best_valid_acc = valid_acc
                count = 0
            else:
                count += 1

        return best_valid_acc

    def predict(self,x):
        # generate the predicted probability of different classes
        hidden_input = x.dot(self.weight_1) + self.bias_1
        hidden_output = tanh(hidden_input)

        output_input = hidden_output.dot(self.weight_2) + self.bias_2
        y = softmax(output_input)
        y = np.argmax(y, axis=1)

        # convert class probability to predicted labels
        return y

    def get_hidden(self,x):
        # extract the intermediate features computed at the hidden layers (after applying activation function)
        hidden_input = x.dot(self.weight_1) + self.bias_1
        hidden_output = tanh(hidden_input)

        return hidden_output

    def params(self):
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2
