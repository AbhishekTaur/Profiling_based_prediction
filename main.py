import torch
from model import MLP
import torch.nn as nn
import pandas as pd
from data_utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)


features = ['Normalized integer', 'Normalized floating', 'Normalized control', 'Normalized time avg',
            'Ratio Memory', 'Ratio branches', 'Ratio call']


one_hot_encoder = {0: [0, 0, 0, 0, 0, 0, 0, 1], 1: [0, 0, 0, 0, 0, 0, 1, 0], 2: [0, 0, 0, 0, 0, 1, 0, 0],
                   3: [0, 0, 0, 0, 1, 0, 0, 0],
                   4: [0, 0, 0, 1, 0, 0, 0, 0], 5: [0, 0, 1, 0, 0, 0, 0, 0], 6: [0, 1, 0, 0, 0, 0, 0, 0],
                   7: [1, 0, 0, 0, 0, 0, 0, 0]}

cores_llc_dict = {0: {'cores': 4, 'llc': 40}, 1: {'cores': 4, 'llc': 60}, 2: {'cores': 4, 'llc': 80},
                  3: {'cores': 4, 'llc': 100}, 4: {'cores': 8, 'llc': 40}, 5: {'cores': 8, 'llc': 60},
                  6: {'cores': 8, 'llc': 80}, 7: {'cores': 8, 'llc': 100}}


def oneHotEncoding(y):
    y_out = []
    for i in range(len(y)):
        y_out.append(one_hot_encoder[y[i]])
    return y_out


def get_data(config_files, n, run_number):
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
    return data_X, data_Y


def get_data_train(n, config_files, run_number):
    data_X, data_Y = get_data(config_files, n, run_number)
    train_size = int(len(data_X)*0.8)
    data_X = data_X[0:train_size]
    data_Y = data_Y[0:train_size]
    X = torch.Tensor(data_X)
    y = torch.Tensor(data_Y.ravel())

    return X, y


def get_data_validate(n, config_files, run_number):
    data_X, data_Y = get_data(config_files, n, run_number)
    data_X = data_X[int(len(data_X) * 0.8):]
    data_Y = data_Y[int(len(data_Y)*0.8):]
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
    for i in range(0, input.size()[0], batch_size):
        optimizer.zero_grad()
        criterion = nn.NLLLoss()

        predicted = model(input[i:i + batch_size])
        predictions.append(np.argmax(predicted.detach().cpu().numpy(), axis=-1))

        loss = criterion(predicted, label[i:i + batch_size])
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

    return np.hstack(tuple(predictions)), np.mean(np.array(losses))


def validate(n, config_files, run_number, model):
    X, y = get_data_validate(n, config_files, run_number)
    X = X.to(device)
    y = y.to(device)
    test_output = []
    with torch.no_grad():
        for i in range(len(X)):
            test_point = X[i]
            output = model(test_point.reshape(1, -1))
            test_output.append(np.argmax(output.detach().cpu().numpy(), axis=-1)[0])
    print('run_number: ', run_number, ',test accuracy: ', np.sum(np.asarray(test_output) == np.asarray(y.cpu())) / len(y))
    print("Frequency output: ", freq(np.array(test_output, dtype=int)))
    print("Frequency y: ", freq(np.array(y.cpu(), dtype=int)))
    return np.sum(np.array(test_output).reshape(-1, 1) == np.array(y.cpu(), dtype=int).reshape(-1, 1)) / len(y)


def test(n, run_number):
    df_4_40 = pd.read_csv('./test_{}/merged_config_test_4_40.csv'.format(run_number))
    df_4_60 = pd.read_csv('./test_{}/merged_config_test_4_60.csv'.format(run_number))
    df_4_80 = pd.read_csv('./test_{}/merged_config_test_4_80.csv'.format(run_number))
    df_4_100 = pd.read_csv('./test_{}/merged_config_test_4_100.csv'.format(run_number))
    df_8_40 = pd.read_csv('./test_{}/merged_config_test_8_40.csv'.format(run_number))
    df_8_60 = pd.read_csv('./test_{}/merged_config_test_8_60.csv'.format(run_number))
    df_8_80 = pd.read_csv('./test_{}/merged_config_test_8_80.csv'.format(run_number))
    df_8_100 = pd.read_csv('./test_{}/merged_config_test_8_100.csv'.format(run_number))
    best_config = pd.read_csv('./test_{}/best_config_file.csv'.format(run_number))
    df_keys = {0: df_4_40, 1: df_4_60, 2: df_4_80, 3: df_4_100,
               4: df_8_40, 5: df_8_60, 6: df_8_80, 7: df_8_100}

    min_rows = 0
    min_rows = get_min_rows(df_keys, min_rows)

    if n == 1:
        model = MLP(15, 16, 8)
    else:
        model = MLP(7, 16, 8)
    model.load_state_dict(torch.load('checkpoint/MLP_model_19_train.pwf', map_location='cpu'))
    model.eval()

    data_point = list(df_8_100.iloc[0, [1, 2, 3, 5, 6, 7, 8]].values)

    if n == 1:
        one_hot_y = [0, 0, 0, 0, 0, 0, 0, 0]
        data_point = torch.Tensor(data_point + one_hot_y)
    else:
        data_point = torch.Tensor(data_point)

    with open("parameters.txt", "w") as f:
        f.write("Parameters \n")
        for i, param in enumerate(list(model.parameters())):
            if i % 2 == 0:
                weight = "weight for {} layer: ".format(i / 2 + 1) + str(param) + "\n"
                f.write(weight)
            else:
                bias = "bias for {} layer: ".format(int(i / 2) + 1) + str(param) + "\n"
                f.write(bias)

    cycles = df_8_100.iloc[0, 4]
    cycles_complete = df_8_100.iloc[0, 4]
    best_cycles = df_keys[best_config.iloc[0, -1]].iloc[0, 4]
    predicted = model.forward(data_point.reshape(1, -1))
    predicted = np.argmax(predicted.detach().cpu().numpy(), axis=-1)
    cycles_array = [int(cycles)]
    cores = [8]
    llc = [100]
    x_pos = [0]
    for i in range(1, min_rows):
        data_point = list(df_keys[predicted[0]].iloc[i, [1, 2, 3, 5, 6, 7, 8]].values)
        if n == 1:
            one_hot_y = oneHotEncoding(predicted)[0]
            data_point = torch.Tensor(data_point + one_hot_y)
        else:
            data_point = torch.Tensor(data_point)
        x_pos.append(cycles)
        cycles_array.append(int(df_keys[predicted[0]].iloc[i, 4]))
        cores.append(cores_llc_dict[predicted[0]]['cores'])
        llc.append(cores_llc_dict[predicted[0]]['llc'])
        cycles = cycles + df_keys[predicted[0]].iloc[i, 4]
        predicted = model.forward(data_point.reshape(1, -1))
        predicted = np.argmax(predicted.detach().cpu().numpy(), axis=-1)
        cycles_complete = cycles_complete + df_8_100.iloc[i, 4]
        best_cycles = best_cycles + df_keys[best_config.iloc[i, -1]].iloc[i, 4]

    print('About to plot the graphs for run_number: {}'.format(run_number))
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 32,
            }

    widths = [cycle * 10**-8*0.8 for cycle in cycles_array]
    x_pos_reduced = [x * 10**-8 for x in x_pos]
    plot_test_results(cores, font, run_number, widths, x_pos_reduced, 'Cores')
    plot_test_results(llc, font, run_number, widths, x_pos_reduced, 'LLC')

    print('run number:', run_number)
    print('cycles calculated:', cycles)
    print('cycles for complete configuration:', cycles_complete)
    print('best configuration cycles:', best_cycles)
    print('complete cycle percentage', cycles/cycles_complete * 100)
    print('best cycle percentage', cycles/best_cycles*100)
    print('\n')


def main():
    [os.remove(os.path.join(".", f)) for f in os.listdir(".") if f.endswith(".png")]
    train_dict = {}
    test_dict = {}
    getConfigFilesList('.', False, 0, train_dict, 'train')
    getConfigFilesList('.', False, 0, test_dict, 'test')
    for key in train_dict.keys():
        train(train_dict[key], key)
        test(1, key)


def train(config_files, run_number):
    max_accuracy = []
    max_validation_accuracy = []
    for n in range(1, 2):
        X, y = get_data_train(n, config_files, run_number)
        input_size, hidden_size, output_size = X.shape[1], 16, 8
        model = MLP(input_size, hidden_size, output_size)
        model.to(device)
        X, y = X.to(device), y.to(device)
        epochs = 20
        accuracy = []
        test_accuracy = []
        for i in range(epochs):
            output_i, loss = train_optim(model, y, X)
            print("epoch {}".format(i))
            print("accuracy = ", np.sum(output_i == y.cpu().numpy()) / y.size())
            print("loss: {}".format(loss))
            accuracy.append((np.sum(output_i == y.cpu().numpy()) / y.size())[0])
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(model.state_dict(), "checkpoint/MLP_model_{}_train.pwf".format(i))
            test_accuracy.append(validate(n, config_files, run_number, model))
            torch.save(model.state_dict(), "checkpoint/MLP_model_{}_validate.pwf".format(i))

        plot_accuracy_n_print(accuracy, max_accuracy, n, run_number, 'train')
        plot_accuracy_n_print(test_accuracy, max_validation_accuracy, n, run_number, 'validate')


if __name__ == "__main__":
    main()
