import os
import re
import numpy as np
import matplotlib.pyplot as plt


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


def plot_accuracy_n_print(accuracy, max_accuracy, n, run_number, mode):
    x = np.arange(len(accuracy))
    plt.plot(x, accuracy)
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    plt.title("{} accuracy over epochs".format(mode))
    max_accuracy.append([max(accuracy), accuracy.index(max(accuracy))])
    plt.savefig('{}_{}_{}.png'.format(mode, run_number, n))
    plt.figure()

    print('run_number: ', run_number, ', Maximum {} accuracy: '.format(mode), max_accuracy)


def freq(lst):
    d = {}
    for i in lst:
        if d.get(i):
            d[i] += 1
        else:
            d[i] = 1
    return


def plot_test_results(x, font, run_number, widths, x_pos_reduced, feature):
    plt.figure(figsize=(25, 15))
    plt.bar(x_pos_reduced, x, width=widths)
    plt.xlabel('Cycles($10^8$)', fontdict=font)
    plt.ylabel('{}'.format(feature), fontdict=font)
    plt.title("{} used over cycles".format(feature), fontdict=font)
    plt.savefig("{}_{}.png".format(feature, run_number))
    plt.close()


def get_min_rows(df_keys, min_rows):
    for i in df_keys.keys():
        rows = df_keys[i].shape[0]
        if i == 0:
            min_rows = rows
        elif min_rows > rows:
            min_rows = rows
    return min_rows
