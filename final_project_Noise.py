
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TwoLayerNN():
    def __init__(self, train, test, hidden_neurons_num=100,  lr=0.1, momentum= 0.9,  max_epoch=50, batch_size= 32):
        '''

        :param hidden_neurons_num:
        :param lr:
        :param momentum:
        :param max_epoch:
        :param num_inputes: 784 +1(bias)
        :param num_outputs: 10 from 0 to 9
        '''
        self.train_dataset = train
        self.test_dataset = test
        self.lr = lr
        self.momentum = momentum
        self.max_epoch = max_epoch
        self.hidden_neurons_num = hidden_neurons_num
        self.num_inputes = 785
        self.num_outputs = 10
        self.batch_size = batch_size
        ## create weights
        self.weights_input = np.random.uniform(-0.05, 0.05, (self.num_inputes, self.hidden_neurons_num))
        self.weights_hidden = np.random.uniform(-0.05, 0.05, (self.hidden_neurons_num +1, self.num_outputs))
        self.past_weights_input = np.zeros((self.num_inputes, self.hidden_neurons_num))
        self.past_weights_hidden = np.zeros((self.hidden_neurons_num +1, self.num_outputs))

        self.hidden_with_bias = np.ones(self.hidden_neurons_num + 1)


    def forward_backward(self, dataset):
        predict_list = []
        actual_list = []
        Ein = np.zeros(dataset.shape[0])
        for data_index in range(dataset.shape[0]):

            target_class = int(dataset[data_index, 0])
            actual_list.append(target_class)
            target = np.ones(10) * 0.1
            target[target_class] = 0.9

            # Input x to the network and compute the activation hj of each hidden unit j.
            self.input_to_hidden = self.sigmoid(np.dot(dataset[data_index, 1:dataset.shape[1]], self.weights_input))
            self.hidden_with_bias[:self.hidden_neurons_num] = self.input_to_hidden
            # Compute the activation  of each output unit k.
            self.hidden_to_output = self.sigmoid(np.dot(self.hidden_with_bias, self.weights_hidden))
            predict_list.append(np.argmax(self.hidden_to_output))


            # Compute the error at the output
            self.error_out = self.hidden_to_output * (1 - self.hidden_to_output) * np.subtract(target, self.hidden_to_output)
            # Compute the error at the hidden layer(s)
            self.error_hidden = self.hidden_with_bias * (1 - self.hidden_with_bias) * np.dot(self.weights_hidden, self.error_out)

            # Update the weights
            delta_weights_hidden = self.lr * np.outer(self.hidden_with_bias,
                                                      self.error_out) + self.momentum * self.past_weights_hidden
            self.past_weights_hidden = delta_weights_hidden
            self.weights_hidden = self.weights_hidden + delta_weights_hidden

            delta_weights_input = self.lr * np.outer(self.error_hidden[:self.hidden_neurons_num], dataset[data_index,
                                                                                             1:self.train_dataset.shape[
                                                                                                 1]]).T + self.momentum * self.past_weights_input

            self.past_weights_input = delta_weights_input
            self.weights_input = self.weights_input + delta_weights_input

            # mu, sigma = 0, 0.1
            # noise = np.random.normal(mu, sigma, [self.weights_input.shape[0], self.weights_input.shape[1]])
            # self.weights_input = self.weights_input + noise

            #Ein[data_index] = np.mean(np.square(self.hidden_to_output - target))
            Ein[data_index] = np.square(np.argmax(self.hidden_to_output) - target_class)
        Ein_final = np.mean(Ein)
        accuracy = (np.array(predict_list) == np.array(actual_list)).sum() / float(len(actual_list)) * 100
        return accuracy, Ein_final


    def sigmoid(self,value):
        return 1/(1 + np.exp(-value))

    def testing(self, dataset, epoch):
        predict_list = []
        actual_list = []
        Ein = np.zeros(dataset.shape[0])
        for data_index in range(dataset.shape[0]):
            target_class = int(dataset[data_index, 0])
            actual_list.append(target_class)
            target = np.ones(10) * 0.1
            target[target_class] = 0.9

            # Input x to the network and compute the activation hj of each hidden unit j.
            self.input_to_hidden = self.sigmoid(np.dot(dataset[data_index, 1:dataset.shape[1]], self.weights_input))
            self.hidden_with_bias[:self.hidden_neurons_num] = self.input_to_hidden
            # Compute the activation  of each output unit k.
            self.hidden_to_output = self.sigmoid(np.dot(self.hidden_with_bias, self.weights_hidden))
            predict_list.append(np.argmax(self.hidden_to_output))

            #Ein[data_index] = np.mean(np.square(self.hidden_to_output - target))
            Ein[data_index] = np.square(np.argmax(self.hidden_to_output) - target_class)
        Ein_final = np.mean(Ein)


        accuracy = (np.array(predict_list) == np.array(actual_list)).sum() / float(len(actual_list)) * 100
        if(epoch == self.max_epoch-1):
            print("Accuracy for test set for one hidden layer with {} hidden unit is:  {}".format(self.hidden_neurons_num, accuracy))
            print("Confusion matrix  on the test set for one hidden layer with {} hidden unit, after training completed at maximum epoch {}".format(self.hidden_neurons_num,epoch))
            print(confusion_matrix(actual_list, predict_list))

        return accuracy, Ein_final

def data_loading(dataset):
    dataset = np.asarray(dataset, dtype=float)
    '''
    MNIST dataset 
    60,000 training examples: shape = (60000, 785)
    10,000 test examples: shape = (10000, 785)
    First value in each row is the target class.
    784 inputs without bias input
    10 output from 0 to 9
    '''
    dataset[:, 1:785] = dataset[:, 1:785] / 255.0
    # Set bias to one
    dataset = np.append(dataset, np.ones((dataset.shape[0], 1)), axis=1)
    return dataset


def main():
    # data loading
    train_data = pd.read_csv('http://www.pjreddie.com/media/files/mnist_train.csv', delimiter=',', header=None).values
    test_data = pd.read_csv('http://www.pjreddie.com/media/files/mnist_test.csv', delimiter=',', header=None).values
    # train_data = np.genfromtxt(r'C:\\Users\asalasow\Desktop\ml\mnist_train.csv', delimiter=',')
    # test_data = pd.read_csv('http://www.pjreddie.com/media/files/mnist_test.csv', delimiter=',')

    hidden_unit = 600
    #for h_unit in list(hidden_unit):
    print ('\n*************** Starting hidden_unit    ',hidden_unit ,'***************')
    plt.figure()
    train = data_loading(test_data)
    test = data_loading(train_data)
    print(train.shape)

    # train = data_loading(train_data)
    # test = data_loading(test_data)

    # print(train.shape)
    # print(train)
    #
    # mu, sigma = 0, 0.1
    # noise = np.random.normal(mu, sigma, [train.shape[0], train.shape[1]])
    # train[:, 1:785] = train[:, 1:785] + noise[:, 1:785]
    # print(train.shape)
    # print(train)

    hidden_neurons_num = hidden_unit
    lr = 0.001
    momentum = 0.0
    max_epoch = 50
    batch_size = 32
    net = TwoLayerNN(train, test, hidden_neurons_num, lr, momentum, max_epoch, batch_size)
    epoch_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    # train_loss = np.zeros(max_epoch)
    # test_loss = np.zeros(max_epoch)

    train_loss = []
    test_loss = []

    for epoch in range(max_epoch):
        print('Starting Epoch', epoch)
        train_accuracy, train_error = net.forward_backward(train)
        test_accuracy, test_error = net.testing(test, epoch)
        epoch_list.append(epoch)
        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)
        # train_loss.append(np.mean(train_error))
        # test_loss.append(np.mean(test_error))

        train_loss.append(train_error)
        test_loss.append(test_error)

        np.random.shuffle(train)

    f = plt.figure(figsize=(20, 10))

    plt.subplot(221, title='Model Loss')
    plt.plot(train_loss, 'r', label='train_loss')
    plt.plot(test_loss, 'b', label='test_loss ')
    plt.xlabel('Epochs')
    plt.ylabel('Error ')
    plt.legend()

    plt.subplot(222, title='Accuracy')
    plt.plot(train_accuracy_list, 'r', label="Training Set")
    plt.plot(test_accuracy_list, 'b', label="Testing Set")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy of dataset (%) ')
    plt.legend()


    plt.show()


if __name__ == "__main__":
    main()