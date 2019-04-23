import torch
import numpy as np
from model import MLP
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

lr = 0.1
seq_length = 20
data_time_steps = np.linspace(2, 100, seq_length + 1)
data = (np.sin(data_time_steps) * 5 + 5).astype(int)
data.resize((seq_length + 1, 1))

features = ['Normalized integer', 'Normalized floating', 'Normalized control', 'Normalized time avg',
            'Ratio Memory', 'Ratio branches', 'Ratio call']

# config_files = ['merged_config_{}_4_40.csv'.format('train'),
#                 'merged_config_{}_4_60.csv'.format('train'),
#                 'merged_config_{}_4_80.csv'.format('train'),
#                 'merged_config_{}_4_100.csv'.format('train'),
#                 'merged_config_{}_8_40.csv'.format('train'),
#                 'merged_config_{}_8_60.csv'.format('train'),
#                 'merged_config_{}_8_80.csv'.format('train'),
#                 'merged_config_{}_8_100.csv'.format('train')]

one_hot_encoder = {0: [0,0,0,0,0,0,0,1], 1: [0,0,0,0,0,0,1,0], 2: [0,0,0,0,0,1,0,0], 3: [0,0,0,0,1,0,0,0],
                   4: [0, 0, 0, 1, 0, 0, 0, 0], 5: [0,0,1,0,0,0,0,0], 6: [0,1,0,0,0,0,0,0], 7: [1,0,0,0,0,0,0,0]}

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

# config_files = ['processed_config_4_40.csv', 'processed_config_4_60.csv', 'processed_config_4_80.csv',
#                 'processed_config_4_100.csv', 'processed_config_8_40.csv', 'processed_config_8_60.csv',
#                 'processed_config_8_80.csv', 'processed_config_8_100.csv']


def get_data(config_files):
    data_X = []
    data_Y = pd.read_csv('best_config_file.csv')
    for config, j in zip(config_files, range(len(config_files))):
        df = pd.read_csv(config, usecols=features).values
        data_X.append(df)
    data_X = np.hstack(tuple(data_X))
    # print(data_X.shape)
    # print(data_Y.shape)
    # X = torch.Tensor(data_X)
    # y = torch.Tensor(data_Y.ravel())
    y_onehot = pd.get_dummies(data_Y.get('Best Configuration')).values[:-1]
    y_onehot = np.vstack((np.zeros(shape=(1, 4), dtype=np.int), y_onehot))
    new_X = np.hstack((data_X, y_onehot))
    X = torch.Tensor(new_X)
    y = torch.Tensor(data_Y.values.ravel())
    return X, y


def get_data_prev_onehot(config_files):
    data_X = []
    data_Y = []
    y_onehot_list = []
    df_y = pd.read_csv('best_config_file.csv')
    y_onehot = pd.get_dummies(df_y.get('Best Configuration')).values[:-1]
    y_onehot = np.vstack((np.zeros(shape=(1, 4), dtype=np.int), y_onehot))
    for config, j in zip(config_files, range(len(config_files))):
        df = pd.read_csv(config, usecols=features).values
        data_X.append(df)
        data_Y.append(df_y.values)
        y_onehot_list.append(y_onehot)
    data_X = np.vstack(tuple(data_X))
    data_Y = np.vstack(tuple(data_Y))
    y_onehot_list = np.vstack(tuple(y_onehot_list))
    data_X = np.hstack((data_X, y_onehot_list))

    X = torch.Tensor(data_X)
    y = torch.Tensor(data_Y.ravel())
    return X, y


def get_data_prev_n(n, config_files, run_number):
    data_X = []
    data_Y = []
    y_onehot_list = []
    df_y = pd.read_csv('train_{}/best_config_file.csv'.format(run_number))
    # number_of_rows = df_y.get('Best Configuration').count()
    if n > 0:
        #y_onehot = pd.get_dummies(df_y.get('Best Configuration')).values[:-n]
        y_onehot = oneHotEncoding(df_y.get('Best Configuration').values)[:-n]
        #y_onehot = np.vstack((np.zeros(shape=(n, y_onehot.shape[1]), dtype=np.int), y_onehot))
        y_onehot = np.vstack((np.zeros(shape=(n, 8), dtype=np.int), y_onehot))
    else:
        y_onehot = pd.get_dummies(df_y.get('Best Configuration')).values[:]
    for config, j in zip(config_files, range(len(config_files))):
        df = pd.read_csv(config, usecols=features).values
        data_X.append(df)
        number_of_rows = len(df/len(features))
        # print(len(data_X))
        #data_Y.append(df_y.get('Best Configuration').values[0:52])

        y_onehot_list.append(y_onehot)
    data_X = np.vstack(tuple(data_X))
    data_Y = df_y.get('Best Configuration').values
    #data_Y = np.vstack(tuple(data_Y))
    #y_onehot_list = np.vstack(tuple(y_onehot_list))
    # data_X = np.array(data_X).reshape(8864, 8)
    if n > 0:
        data_X = np.hstack((data_X, y_onehot))
        #data_X = np.hstack((data_X, y_onehot_list))

    X = torch.Tensor(data_X)
    y = torch.Tensor(data_Y.ravel())
    return X, y


def get_validation_data(n, config_files, run_number):
    data_X = []
    data_Y = []
    df_y = pd.read_csv('test_{}/best_config_file.csv'.format(run_number))
    one_hot = 0
    if n > 0:
        y_onehot = pd.get_dummies(df_y.get('Best Configuration')).values
        one_hot = y_onehot.shape[1]
    for config, j in zip(config_files, range(len(config_files))):
        df = pd.read_csv(config, usecols=features).values
        data_X.append(df)
    data_X = np.vstack(tuple(data_X))
    data_Y = df_y.get('Best Configuration').values

    return data_X, data_Y.ravel(), one_hot


def get_data_prev(config_files):
    data_X = np.array([[]])
    data_Y = np.array([[]])
    for config, j in zip(config_files, range(len(config_files))):
        df = pd.read_csv(config, usecols=features)
        df_y = pd.read_csv('best_config_file.csv')
        for i in range(df['Phase'].size):
            data_X = np.append([data_X], df.iloc[i])
            data_Y = np.append(data_Y, df_y.iloc[i])
    data_X = data_X.reshape(8864, 8)
    X = torch.Tensor(data_X)
    y = torch.Tensor(data_Y.ravel())
    return X, y


def train_RNN(model, label, input):
    label = label.long()
    # print(type(label))
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

            # Add parameters' gradients to their values, multiplied by learning rate
            for p in model.parameters():
                if p.grad is not None:
                    p.data.add_(-learning_rate, p.grad.data)

    return np.vstack(tuple(predictions)), np.mean(np.array(losses))


# def train(model, label, input):
#     label = label.long()
#     batch_size = 32
#     # learning_rate = 0.00005
#     learning_rate = 0.0001
#     predictions = []
#     losses = []
#     for i in range(0, input.size()[0], batch_size):
#
#         model.zero_grad()
#         criterion = nn.NLLLoss()
#
#         predicted = model(input[i:i + batch_size])
#         predictions.append(np.argmax(predicted.detach().numpy(), axis=-1))
#
#         loss = criterion(predicted, label[i:i + batch_size])
#         loss.backward()
#         losses.append(loss.item())
#
#         # Add parameters' gradients to their values, multiplied by learning rate
#         for p in model.parameters():
#             p.data.add_(-learning_rate, p.grad.data)
#
#     return np.hstack(tuple(predictions)), np.mean(np.array(losses))


def train_optim(model, label, input):
    label = label.long()
    batch_size = 32
    # learning_rate = 0.00005
    learning_rate = 0.1
    predictions = []
    losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for i in range(0, input.size()[0], batch_size):
        optimizer.zero_grad()
        criterion = nn.NLLLoss()

        predicted = model(input[i:i + batch_size])
        predictions.append(np.argmax(predicted.detach().numpy(), axis=-1))

        loss = criterion(predicted, label[i:i + batch_size])
        loss.backward()
        losses.append(loss.item())

        # Add parameters' gradients to their values, multiplied by learning rate
        # for p in model.parameters():
        #     p.data.add_(-learning_rate, p.grad.data)
        optimizer.step()

    return np.hstack(tuple(predictions)), np.mean(np.array(losses))


def validate(n, test_files, run_number):
    # input_size, hidden_size, output_size = 12, 16, 8
    # model = MLP(input_size, hidden_size, output_size)
    X, y, one_hot = get_validation_data(n, test_files, run_number)
    input_size, hidden_size, output_size = X.shape[1] + one_hot, 16, 8
    model = MLP(input_size, hidden_size, output_size)
    test_output = []
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
    print('run_number: ', run_number, ',test accuracy: ', np.sum(test_output == np.asarray(y))/ y.size)


def main():
    # input_size, hidden_size, output_size = 68, 128, 8
    # input_size, hidden_size, output_size = 12, 16, 8

    # X, y = get_data()
    # X, y = get_data_prev()
    # X, y = get_data_prev_onehot()
    train_dict = {}
    test_dict = {}
    getConfigFilesList('.', False, 0, train_dict, 'train')
    getConfigFilesList('.', False, 0, test_dict, 'test')
    for key in train_dict.keys():
        # print(train_dict[key])
        train(train_dict[key], key, test_dict[key])
    #test()
    # model = RNN(input_size, hidden_size, output_size)
    # epochs = 500
    # print(y.size())
    # for i in range(epochs):
    #     output_i, loss = train_RNN(model, y, X)
    #     print("epoch {}".format(i))
    #     print("accuracy = ", np.sum(output_i == y.numpy()) / y.size())
    #     print("loss: {}".format(loss))

    # for i in range(X.size(0)):
    #     input = X[i:i + 1]
    #     (pred, context_state) = model.forward(input, context_state, model.w1, model.w2)
    #     context_state = context_state
    #     predictions.append(pred.data.numpy().ravel()[0])
    #
    # pl.scatter(data_time_steps[:-1], X.data.numpy(), s=90, label="Actual")
    # pl.scatter(data_time_steps[1:], predictions, label="Predicted")
    # pl.legend()
    # pl.show()


def train(config_files, run_number, test_files):
    max_accuracy = []
    # input_size, hidden_size, output_size = 12, 16, 8
    # model = MLP(input_size, hidden_size, output_size)
    for n in range(0, 3):
        X, y = get_data_prev_n(n, config_files, run_number)
        # if n == 0:
        #     input_size, hidden_size, output_size = 7, 16, 8
        # else:
        #     input_size, hidden_size, output_size = 10, 16, 8
        input_size, hidden_size, output_size = X.shape[1], 16, 8
        model = MLP(input_size, hidden_size, output_size)
        epochs = 2
        accuracy = []
        for i in range(epochs):
            # output_i, loss = train(model, y, X)
            output_i, loss = train_optim(model, y, X)
            print("epoch {}".format(i))
            print("accuracy = ", np.sum(output_i == y.numpy()) / y.size())
            print("loss: {}".format(loss))
            accuracy.append((np.sum(output_i == y.numpy()) / y.size())[0])
            validate(n, test_files, run_number)

        x = np.arange(len(accuracy))
        plt.bar(x, height=accuracy, align='center')
        plt.xlabel("epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over epochs")

        max_accuracy.append([max(accuracy), accuracy.index(max(accuracy))])
        plt.savefig('train_{}_{}.png'.format(run_number, n))
        plt.figure()
    print('run_number: ', run_number, ', Maximum accuracy: ', max_accuracy)


if __name__ == "__main__":
    main()
