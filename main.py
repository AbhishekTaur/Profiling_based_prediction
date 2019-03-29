import torch
from torch.autograd import Variable
import numpy as np
from model import RNN_model
import matplotlib.pyplot as pl
import torch.nn as nn
import pandas as pd


lr = 0.1
seq_length = 20
data_time_steps = np.linspace(2, 100, seq_length + 1)
data = (np.sin(data_time_steps) * 5 + 5).astype(int)
data.resize((seq_length + 1, 1))

features = ['Normalized integer', 'Normalized floating', 'Normalized control', 'Normalized time avg',
           'Ratio Memory', 'Ratio branches', 'Ratio call', 'Phase']

config_files = ['processed_config_4_40.csv', 'processed_config_4_60.csv', 'processed_config_4_80.csv',
                'processed_config_4_100.csv', 'processed_config_8_40.csv', 'processed_config_8_60.csv',
                'processed_config_8_80.csv', 'processed_config_8_100.csv']


def get_data(dtype):
    data_X = np.array([[]])
    data_Y = np.array([[]])
    for config, j in zip(config_files, range(len(config_files))):
        df = pd.read_csv(config, usecols=features)
        df_y = pd.read_csv('best_config_file.csv')
        for i in range(df['Phase'].size):
            data_X = np.append([data_X], df.iloc[i])
            data_Y = np.append(data_Y, df_y.iloc[i])
    data_X = data_X.reshape(8864, 8)
    X = Variable(torch.Tensor(data_X[:-1]).type(dtype), requires_grad=False)
    y = Variable(torch.Tensor(data_Y[:-1].ravel()).type(dtype), requires_grad=False)
    return X, y


def train(X, y, model):
    epochs = 300
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()

    for i in range(epochs):
        total_loss = 0
        context_state = Variable(torch.zeros((model.hidden_size, 8)).type(model.dtype), requires_grad=True)

        for j in range(X.size(0)):
            input = X[j:(j + 1)]
            target = y[j:(j + 1)]

            print(input.shape)
            print(context_state.shape)

            (pred, context_state) = model.forward(input, context_state, model.w1, model.w2)
            # print("pred ", pred.item())
            # print(type(pred))
            label = target.long()
            # print(type(target))

            loss = criterion(pred, label)
            loss.backward()
            model.w1.data -= lr * model.w1.grad.data
            model.w2.data -= lr * model.w2.grad.data
            model.w1.grad.data.zero_()
            model.w2.grad.data.zero_()

            total_loss += loss.item()
            context_state = Variable(context_state.data)
        if i % 10 == 0:
            print("Epoch: {} loss {}".format(i, total_loss))


def main():
    dtype = torch.FloatTensor
    input_size, hidden_size, output_size = 7, 6, 10
    model = RNN_model(input_size, hidden_size, output_size, dtype)
    X, y = get_data(dtype)
    labels = np.array([1, 2, 3, 4, 5])
    train(X, y, model)
    context_state = Variable(torch.zeros((1, model.hidden_size)).type(model.dtype), requires_grad=False)
    predictions = []

    for i in range(X.size(0)):
        input = X[i:i + 1]
        (pred, context_state) = model.forward(input, context_state, model.w1, model.w2)
        context_state = context_state
        predictions.append(pred.data.numpy().ravel()[0])

    pl.scatter(data_time_steps[:-1], X.data.numpy(), s=90, label="Actual")
    pl.scatter(data_time_steps[1:], predictions, label="Predicted")
    pl.legend()
    pl.show()


if __name__ == "__main__":
    main()