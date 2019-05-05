import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from model import MLP
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from collections import Counter


lr = 0.1
seq_length = 20
data_time_steps = np.linspace(2, 100, seq_length + 1)
data = (np.sin(data_time_steps) * 5 + 5).astype(int)
data.resize((seq_length + 1, 1))

features = ['Normalized integer', 'Normalized floating', 'Normalized control', 'Normalized time avg',
            'Ratio Memory', 'Ratio branches', 'Ratio call']

one_hot_encoder = {0: [0, 0, 0, 0, 0, 0, 0, 1], 1: [0, 0, 0, 0, 0, 0, 1, 0], 2: [0, 0, 0, 0, 0, 1, 0, 0],
                   3: [0, 0, 0, 0, 1, 0, 0, 0],
                   4: [0, 0, 0, 1, 0, 0, 0, 0], 5: [0, 0, 1, 0, 0, 0, 0, 0], 6: [0, 1, 0, 0, 0, 0, 0, 0],
                   7: [1, 0, 0, 0, 0, 0, 0, 0]}


def oneHotEncoding(y):
    y_out = []
    for i in range(len(y)):
        y_out.append(one_hot_encoder[y[i]])
    return y_out


def getConfigFilesList(dirName, inside, run_number, dict_t, phase):
    listOfFile = os.listdir(dirName)
    for entry in listOfFile:
        if phase == 'train':
            searchString = '^train_\d{1,4}$'
        else:
            searchString = '^test_\d{1,4}$'
        if re.search(searchString, entry):
            inside = True
            run_number = int(entry.split("_")[1])
            if not run_number in dict_t.keys():
                dict_t[run_number] = []
            elif len(dict_t[run_number]) == 8:
                inside = False

        if inside:
            fullPath = os.path.join(dirName, entry)
            if os.path.isdir(fullPath):
                getConfigFilesList(fullPath, inside, run_number, dict_t, phase)
            elif 'merged_config_{}'.format(phase) in fullPath:
                dict_t[run_number].append(fullPath)


def get_data_prev_n(n, config_files, run_number):
    data_X = []
    y_onehot_list = []
    df_y = pd.read_csv('train_{}/best_config_file.csv'.format(run_number))
    if n > 0:
        y_onehot = oneHotEncoding(df_y.get('Best Configuration').values)[:-n]
        y_onehot = np.vstack((np.zeros(shape=(n, 8), dtype=np.int), y_onehot))
    else:
        y_onehot = pd.get_dummies(df_y.get('Best Configuration')).values[:]
    for config, j in zip(config_files, range(len(config_files))):
        df = pd.read_csv(config, usecols=features).values
        data_X.append(df)

        y_onehot_list.append(y_onehot)
    data_X = np.vstack(tuple(data_X))
    data_Y = df_y.get('Best Configuration').values
    if n > 0:
        data_X = np.hstack((data_X, y_onehot))

    scaler = MinMaxScaler()
    scaler.fit(data_X)
    data_X = scaler.transform(data_X)
    # p = np.random.permutation(len(data_X))
    # X = torch.Tensor(data_X[p])
    # y = torch.Tensor(data_Y[p].ravel())
    # indexes = parse_indexes(data_Y)
    # np.random.shuffle(data_X[indexes])
    # np.random.shuffle(data_Y[indexes].ravel())
    # X = torch.Tensor(data_X[indexes])
    # y = torch.Tensor(data_Y[indexes].ravel())
    np.random.shuffle(data_X)
    np.random.shuffle(data_Y)
    X = torch.Tensor(data_X)
    y = torch.Tensor(data_Y.ravel())
    return X, y


def get_validation_data(n, config_files, run_number):
    data_X = []
    df_y = pd.read_csv('test_{}/best_config_file.csv'.format(run_number))
    one_hot = 0
    if n > 0:
        y_onehot = oneHotEncoding(df_y.get('Best Configuration').values)
        one_hot = len(y_onehot[0])
    for config, j in zip(config_files, range(len(config_files))):
        df = pd.read_csv(config, usecols=features).values
        data_X.append(df)
    data_X = np.vstack(tuple(data_X))
    data_Y = df_y.get('Best Configuration').values

    return data_X, data_Y.ravel(), one_hot


def train_RNN(model, label, input):
    label = label.long()
    learning_rate = 0.0001
    predictions = []
    losses = []
    criterion = nn.NLLLoss()
    for i in range(input.size()[0]):
        hidden = model.initHidden()
        model.zero_grad()

        predicted, hidden = model(input[i].reshape(1, -1), hidden)
        predictions.append(np.argmax(predicted.detach().numpy(), axis=-1))

        if i % 4 == 0:
            loss = criterion(predicted, label[i:i + 1])
            loss.backward(retain_graph=True)
            losses.append(loss.item())

            for p in model.parameters():
                if p.grad is not None:
                    p.data.add_(-learning_rate, p.grad.data)

    return np.vstack(tuple(predictions)), np.mean(np.array(losses))


def train_optim(model, label, input):
    label = label.long()
    batch_size = 32
    predictions = []
    losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # print("Minumum: ", np.min(input.numpy(), axis=1))
    # print("Maximum: ", np.max(input.numpy(), axis=1))
    for i in range(0, input.size()[0], batch_size):
        optimizer.zero_grad()
        criterion = nn.NLLLoss()

        predicted = model(input[i:i + batch_size])
        predictions.append(np.argmax(predicted.detach().numpy(), axis=-1))

        loss = criterion(predicted, label[i:i + batch_size])
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

    return np.hstack(tuple(predictions)), np.mean(np.array(losses))


def validate(n, test_files, run_number, model):
    X, y, one_hot = get_validation_data(n, test_files, run_number)
    test_output = []
    with torch.no_grad():
        for i in range(len(X)):
            test_point = X[i].ravel()
            if i == 0:
                if n > 0:
                    test_point = np.hstack((test_point.reshape(1, test_point.shape[0]), np.zeros(shape=(1, 8))))
            else:
                if n > 0:
                    test_point = np.hstack((test_point.reshape(1, test_point.shape[0]), oneHotEncoding([test_output[-1]])))

            output = model(torch.Tensor(test_point).reshape(1, -1))

            test_output.append(np.argmax(output.detach().numpy(), axis=-1)[0])
    print('run_number: ', run_number, ',test accuracy: ', np.sum(np.asarray(test_output) == np.asarray(y)) / y.size)
    print("Frequency output: ", freq(test_output))
    print("Frequency y: ", freq(y))
    results = confusion_matrix(y, test_output)
    print('Confusion Matrix :')
    print(results)
    # print('Accuracy Score :', accuracy_score(y, test_output))
    return np.sum(test_output == np.asarray(y)) / y.size


def main():
    [os.remove(os.path.join(".", f)) for f in os.listdir(".") if f.endswith(".png")]
    train_dict = {}
    test_dict = {}
    getConfigFilesList('.', False, 0, train_dict, 'train')
    getConfigFilesList('.', False, 0, test_dict, 'test')
    for key in train_dict.keys():
        train(train_dict[key], key, test_dict[key])


def plot_output_epoch(output, i, n):
    x = np.arange(len(output))
    plt.plot(x, height=output, align='center')
    plt.xlabel("datapoint")
    plt.ylabel("Predicted")
    plt.title("Prediction for {} epoch".format(i))

    plt.savefig('Predicted_{}_{}.png'.format(i, n))
    plt.figure()


def plot_actual_epoch(output, i, n):
    x = np.arange(len(output))
    plt.plot(x, height=output, align='center')
    plt.xlabel("datapoint")
    plt.ylabel("Predicted")
    plt.title("Actual for {} epoch".format(i))

    plt.savefig('Actual_{}_{}.png'.format(i, n))
    plt.figure()


def freq(lst):
    d = {}
    for i in lst:
        if d.get(i):
            d[i] += 1
        else:
            d[i] = 1
    return d


def parse_indexes(list1):
    index_list = []
    dictor = Counter(list1)
    counter_list = [dictor[x] for x in dictor]
    counter_list.sort()
    if len(counter_list) > 1:
        indexes = counter_list[-2]

    for x in dictor:
        index_list = index_list + list(np.where(list1 == x)[0][0:indexes])
    return index_list


def train(config_files, run_number, test_files):
    max_accuracy = []
    max_validation_accuracy = []
    for n in range(0, 2):
        X, y = get_data_prev_n(n, config_files, run_number)
        input_size, hidden_size, output_size = X.shape[1], 8, 8
        model = MLP(input_size, hidden_size, output_size)
        epochs = 200
        accuracy = []
        test_accuracy = []
        for i in range(epochs):
            output_i, loss = train_optim(model, y, X)
            print("epoch {}".format(i))
            print("accuracy = ", np.sum(output_i == y.numpy()) / y.size())
            print("loss: {}".format(loss))
            accuracy.append((np.sum(output_i == y.numpy()) / y.size())[0])
            test_accuracy.append(validate(n, test_files, run_number, model))

        x = np.arange(len(accuracy))
        plt.plot(x, accuracy)
        plt.xlabel("epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over epochs")

        max_accuracy.append([max(accuracy), accuracy.index(max(accuracy))])
        plt.savefig('train_{}_{}.png'.format(run_number, n))
        plt.figure()

        x_validate = np.arange(len(test_accuracy))
        plt.plot(x_validate, test_accuracy, color='r')
        plt.xlabel("epochs")
        plt.ylabel("Accuracy")
        plt.title("Test Accuracy over epochs")

        max_validation_accuracy.append([max(test_accuracy), test_accuracy.index(max(test_accuracy))])
        plt.savefig('validate_{}_{}.png'.format(run_number, n))
        plt.figure()

    print('run_number: ', run_number, ', Maximum accuracy: ', max_accuracy)
    print('run_number: ', run_number, ', Maximum validation accuracy: ', max_validation_accuracy)


if __name__ == "__main__":
    main()
