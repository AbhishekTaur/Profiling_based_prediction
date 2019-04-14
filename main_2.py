import torch
import numpy as np
from model import MLP, RNN
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

lr = 0.1
seq_length = 20
data_time_steps = np.linspace(2, 100, seq_length + 1)
data = (np.sin(data_time_steps) * 5 + 5).astype(int)
data.resize((seq_length + 1, 1))

features = ['Normalized integer', 'Normalized floating', 'Normalized control', 'Normalized time avg',
            'Ratio Memory', 'Ratio branches', 'Ratio call', 'Phase']

config_files = ['merged_config_4_40.csv', 'merged_config_4_60.csv', 'merged_config_4_80.csv',
                'merged_config_4_100.csv', 'merged_config_8_40.csv', 'merged_config_8_60.csv',
                'merged_config_8_80.csv', 'merged_config_8_100.csv']

# config_files = ['processed_config_4_40.csv', 'processed_config_4_60.csv', 'processed_config_4_80.csv',
#                 'processed_config_4_100.csv', 'processed_config_8_40.csv', 'processed_config_8_60.csv',
#                 'processed_config_8_80.csv', 'processed_config_8_100.csv']


def get_data():
    data_X = []
    data_Y = pd.read_csv('best_config_file.csv')
    for config, j in zip(config_files, range(len(config_files))):
        df = pd.read_csv(config, usecols=features).values
        data_X.append(df)
    data_X = np.hstack(tuple(data_X))
    print(data_X.shape)
    print(data_Y.shape)
    # X = torch.Tensor(data_X)
    # y = torch.Tensor(data_Y.ravel())
    y_onehot = pd.get_dummies(data_Y.get('Best Configuration')).values[:-1]
    y_onehot = np.vstack((np.zeros(shape=(1, 4), dtype=np.int), y_onehot))
    new_X = np.hstack((data_X, y_onehot))
    X = torch.Tensor(new_X)
    y = torch.Tensor(data_Y.values.ravel())
    return X, y


def get_data_prev_onehot():
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


def get_data_prev_n(n):
    data_X = []
    data_Y = []
    y_onehot_list = []
    df_y = pd.read_csv('best_config_file.csv')
    if n > 0:
        y_onehot = pd.get_dummies(df_y.get('Best Configuration')).values[:-n]
        print(y_onehot.shape)
        y_onehot = np.vstack((np.zeros(shape=(n, 4), dtype=np.int), y_onehot))
    else:
        y_onehot = pd.get_dummies(df_y.get('Best Configuration')).values[:]
    for config, j in zip(config_files, range(len(config_files))):
        df = pd.read_csv(config, usecols=features).values
        data_X.append(df)
        data_Y.append(df_y.get('Best Configuration').values)
        y_onehot_list.append(y_onehot)
    data_X = np.vstack(tuple(data_X))
    data_Y = np.vstack(tuple(data_Y))
    y_onehot_list = np.vstack(tuple(y_onehot_list))
    # data_X = np.array(data_X).reshape(8864, 8)
    if n > 0:
        data_X = np.hstack((data_X, y_onehot_list))

    X = torch.Tensor(data_X)
    y = torch.Tensor(data_Y.ravel())
    return X, y


def get_data_prev():
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
    print(type(label))
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


def train(model, label, input):
    label = label.long()
    batch_size = 32
    # learning_rate = 0.00005
    learning_rate = 0.0001
    predictions = []
    losses = []
    for i in range(0, input.size()[0], batch_size):

        model.zero_grad()
        criterion = nn.NLLLoss()

        predicted = model(input[i:i + batch_size])
        predictions.append(np.argmax(predicted.detach().numpy(), axis=-1))

        loss = criterion(predicted, label[i:i + batch_size])
        loss.backward()
        losses.append(loss.item())

        # Add parameters' gradients to their values, multiplied by learning rate
        for p in model.parameters():
            p.data.add_(-learning_rate, p.grad.data)

    return np.hstack(tuple(predictions)), np.mean(np.array(losses))


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


def main():
    # input_size, hidden_size, output_size = 68, 128, 8
    # input_size, hidden_size, output_size = 12, 16, 8

    # X, y = get_data()
    # X, y = get_data_prev()
    # X, y = get_data_prev_onehot()
    max_accuracy = []
    # input_size, hidden_size, output_size = 12, 16, 8
    # model = MLP(input_size, hidden_size, output_size)
    for n in range(1, 5):
        if n == 0:
            input_size, hidden_size, output_size = 8, 16, 8
        else:
            input_size, hidden_size, output_size = 12, 16, 8
        model = MLP(input_size, hidden_size, output_size)
        X, y = get_data_prev_n(n)
        epochs = 500
        accuracy = []
        print(y.size())
        for i in range(epochs):
            # output_i, loss = train(model, y, X)
            output_i, loss = train_optim(model, y, X)
            print("epoch {}".format(i))
            print("accuracy = ", np.sum(output_i == y.numpy()) / y.size())
            print("loss: {}".format(loss))
            accuracy.append((np.sum(output_i == y.numpy()) / y.size())[0])

        x = np.arange(len(accuracy))
        plt.bar(x, height=accuracy, align='center')
        plt.xlabel("epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over epochs")

        max_accuracy.append([max(accuracy), accuracy.index(max(accuracy))])
        plt.savefig('train_{}.png'.format(n))
        plt.figure()

    print(max_accuracy)
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


if __name__ == "__main__":
    main()
