import os
import numpy as np
import re
import datetime
import pandas as pd
from random import sample
from random import randint
from random import seed

program_dict = {'blackscholes': 0, 'dedup': 1, 'streamcluster': 2, 'swaptions': 3, 'freqmine': 4, 'fluidanimate': 5,
                'canneal': 6}

seed(100)


def getListOfFiles(dirName, subset, train):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        if dirName == '../data' and train and program_dict[entry] not in subset:
            continue
        if dirName == '../data' and not train and program_dict[entry] in subset:
            continue
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath, subset, train)
        else:
            allFiles.append(fullPath)

    return allFiles


def getLeastNumberOfRows(merge_dict):
    min_rows = min([len(merge_dict[key]) for key in merge_dict.keys()])
    print(min_rows)
    for key in merge_dict.keys():
        merge_dict[key] = merge_dict[key][0:min_rows]


def merge_data(processed_file, merged_file):
    df = pd.read_csv(processed_file)
    merge_dict = {}
    for key in df.keys():
        merge_dict[key] = []
        count = 0
        for file in df['Filename']:
            if "mem" in file and key == 'Normalized time avg':
                merge_dict[key].append(df[key].values[count])
            elif re.search("x86.", file) and key != 'Normalized time avg':
                merge_dict[key].append(df[key].values[count])
            count = count + 1
    getLeastNumberOfRows(merge_dict)

    df_merge = pd.DataFrame(data=merge_dict)
    df_merge.to_csv(path_or_buf='./{}'.format(merged_file), index=False)


def main():
    start = datetime.datetime.now()
    print("Start time: ", start)
    dirName = '../data'
    print("Starting the program")

    # Get the list of all files in directory tree at given path

    subset = sample([i for i in range(6)], 5)
    listOfTrainFiles = sorted(getListOfFiles(dirName, subset, True))
    listOfTestFiles = sorted(getListOfFiles(dirName, subset, False))
    run_number = str(randint(0, 10000))
    train_data_folder = 'train_{}'.format(run_number)
    test_data_folder = 'test_{}'.format(run_number)
    if not os.path.exists(train_data_folder):
        os.mkdir(test_data_folder)
        os.mkdir(train_data_folder)
        print("Created directories: {}".format(train_data_folder))
        print("Created directories: {}".format(test_data_folder))

    row = ['Filename', 'Normalized integer', 'Normalized floating', 'Normalized control', 'Cycles',
           'Normalized time avg',
           'Ratio Memory', 'Ratio branches', 'Ratio call', 'Phase']
    config_train_files = ['{}/processed_config_{}_4_40.csv'.format(train_data_folder, 'train'),
                          '{}/processed_config_{}_4_60.csv'.format(train_data_folder, 'train'),
                          '{}/processed_config_{}_4_80.csv'.format(train_data_folder, 'train'),
                          '{}/processed_config_{}_4_100.csv'.format(train_data_folder, 'train'),
                          '{}/processed_config_{}_8_40.csv'.format(train_data_folder, 'train'),
                          '{}/processed_config_{}_8_60.csv'.format(train_data_folder, 'train'),
                          '{}/processed_config_{}_8_80.csv'.format(train_data_folder, 'train'),
                          '{}/processed_config_{}_8_100.csv'.format(train_data_folder, 'train')]
    config_test_files = ['{}/processed_config_{}_4_40.csv'.format(test_data_folder, 'test'),
                         '{}/processed_config_{}_4_60.csv'.format(test_data_folder, 'test'),
                         '{}/processed_config_{}_4_80.csv'.format(test_data_folder, 'test'),
                         '{}/processed_config_{}_4_100.csv'.format(test_data_folder, 'test'),
                         '{}/processed_config_{}_8_40.csv'.format(test_data_folder, 'test'),
                         '{}/processed_config_{}_8_60.csv'.format(test_data_folder, 'test'),
                         '{}/processed_config_{}_8_80.csv'.format(test_data_folder, 'test'),
                         '{}/processed_config_{}_8_100.csv'.format(test_data_folder, 'test')]

    merged_train_files = ['{}/merged_config_{}_4_40.csv'.format(train_data_folder, 'train'),
                          '{}/merged_config_{}_4_60.csv'.format(train_data_folder, 'train'),
                          '{}/merged_config_{}_4_80.csv'.format(train_data_folder, 'train'),
                          '{}/merged_config_{}_4_100.csv'.format(train_data_folder, 'train'),
                          '{}/merged_config_{}_8_40.csv'.format(train_data_folder, 'train'),
                          '{}/merged_config_{}_8_60.csv'.format(train_data_folder, 'train'),
                          '{}/merged_config_{}_8_80.csv'.format(train_data_folder, 'train'),
                          '{}/merged_config_{}_8_100.csv'.format(train_data_folder, 'train')]
    merged_test_files = ['{}/merged_config_{}_4_40.csv'.format(test_data_folder, 'test'),
                         '{}/merged_config_{}_4_60.csv'.format(test_data_folder, 'test'),
                         '{}/merged_config_{}_4_80.csv'.format(test_data_folder, 'test'),
                         '{}/merged_config_{}_4_100.csv'.format(test_data_folder, 'test'),
                         '{}/merged_config_{}_8_40.csv'.format(test_data_folder, 'test'),
                         '{}/merged_config_{}_8_60.csv'.format(test_data_folder, 'test'),
                         '{}/merged_config_{}_8_80.csv'.format(test_data_folder, 'test'),
                         '{}/merged_config_{}_8_100.csv'.format(test_data_folder, 'test')]
    create_data(config_train_files, listOfTrainFiles, merged_train_files, row, 'train', run_number)
    create_data(config_test_files, listOfTestFiles, merged_test_files, row, 'test', run_number)

    write_best_config(merged_train_files, run_number, 'train')
    write_best_config(merged_test_files, run_number, 'test')
    end = datetime.datetime.now()
    print("End time: ", end)
    total_time = end - start
    print("Total time in seconds: ", int(total_time.total_seconds()))


def return_dict():
    return {'Filename': [], 'Normalized integer': [], 'Normalized floating': [], 'Normalized control': [], 'Cycles': [],
            'Normalized time avg': [],
            'Ratio Memory': [], 'Ratio branches': [], 'Ratio call': [], 'Phase': []}


def create_data(config_files, listOfTrainFiles, merged_files, row, phase, run_number):
    process_df_4_100_train = return_dict()
    process_df_4_80_train = return_dict()
    process_df_4_60_train = return_dict()
    process_df_4_40_train = return_dict()
    process_df_8_100_train = return_dict()
    process_df_8_80_train = return_dict()
    process_df_8_60_train = return_dict()
    process_df_8_40_train = return_dict()
    process_df_4_100_test = return_dict()
    process_df_4_80_test = return_dict()
    process_df_4_60_test = return_dict()
    process_df_4_40_test = return_dict()
    process_df_8_100_test = return_dict()
    process_df_8_80_test = return_dict()
    process_df_8_60_test = return_dict()
    process_df_8_40_test = return_dict()
    exists = False
    for config, merge_file in zip(config_files, merged_files):
        exists = os.path.exists(config)
    if not exists:
        for elem in listOfTrainFiles:

            file = open(elem, "r")
            if '4core-100cache' in file.name:
                if phase == 'train':
                    process_df_4_100_train = process_file(file, process_df_4_100_train, row)
                else:
                    process_df_4_100_test = process_file(file, process_df_4_100_test, row)
            if '4core-80cache' in file.name:
                if phase == 'train':
                    process_df_4_80_train = process_file(file, process_df_4_80_train, row)
                else:
                    process_df_4_80_test = process_file(file, process_df_4_80_test, row)
            if '4core-60cache' in file.name:
                if phase == 'train':
                    process_df_4_60_train = process_file(file, process_df_4_60_train, row)
                else:
                    process_df_4_60_test = process_file(file, process_df_4_60_test, row)
            if '4core-40cache' in file.name:
                if phase == 'train':
                    process_df_4_40_train = process_file(file, process_df_4_40_train, row)
                else:
                    process_df_4_40_test = process_file(file, process_df_4_40_test, row)
            if '8core-100cache' in file.name:
                if phase == 'train':
                    process_df_8_100_train = process_file(file, process_df_8_100_train, row)
                else:
                    process_df_8_100_test = process_file(file, process_df_8_100_test, row)
            if '8core-80cache' in file.name:
                if phase == 'train':
                    process_df_8_80_train = process_file(file, process_df_8_80_train, row)
                else:
                    process_df_8_80_test = process_file(file, process_df_8_80_test, row)
            if '8core-60cache' in file.name:
                if phase == 'train':
                    process_df_8_60_train = process_file(file, process_df_8_60_train, row)
                else:
                    process_df_8_60_test = process_file(file, process_df_8_60_test, row)
            if '8core-40cache' in file.name:
                if phase == 'train':
                    process_df_8_40_train = process_file(file, process_df_8_40_train, row)
                else:
                    process_df_8_40_test = process_file(file, process_df_8_40_test, row)
            file.close()
    for config_file in config_files:
        if '4_100' in config_file:
            if phase == 'train':
                pd.DataFrame(data=process_df_4_100_train).to_csv(config_file, index=False)
            else:
                pd.DataFrame(data=process_df_4_100_test).to_csv(config_file, index=False)
        if '4_80' in config_file:
            if phase == 'train':
                pd.DataFrame(data=process_df_4_80_train).to_csv(config_file, index=False)
            else:
                pd.DataFrame(data=process_df_4_80_test).to_csv(config_file, index=False)
        if '4_60' in config_file:
            if phase == 'train':
                pd.DataFrame(data=process_df_4_60_train).to_csv(config_file, index=False)
            else:
                pd.DataFrame(data=process_df_4_60_test).to_csv(config_file, index=False)
        if '4_40' in config_file:
            if phase == 'train':
                pd.DataFrame(data=process_df_4_40_train).to_csv(config_file, index=False)
            else:
                pd.DataFrame(data=process_df_4_40_test).to_csv(config_file, index=False)
        if '8_100' in config_file:
            if phase == 'train':
                pd.DataFrame(data=process_df_8_100_train).to_csv(config_file, index=False)
            else:
                pd.DataFrame(data=process_df_8_100_test).to_csv(config_file, index=False)
        if '8_80' in config_file:
            if phase == 'train':
                pd.DataFrame(data=process_df_8_80_train).to_csv(config_file, index=False)
            else:
                pd.DataFrame(data=process_df_8_80_test).to_csv(config_file, index=False)
        if '8_60' in config_file:
            if phase == 'train':
                pd.DataFrame(data=process_df_8_60_train).to_csv(config_file, index=False)
            else:
                pd.DataFrame(data=process_df_8_60_test).to_csv(config_file, index=False)
        if '8_40' in config_file:
            if phase == 'train':
                pd.DataFrame(data=process_df_8_40_train).to_csv(config_file, index=False)
            else:
                pd.DataFrame(data=process_df_8_40_test).to_csv(config_file, index=False)
    for processed_file, merge_file in zip(config_files, merged_files):
        if not os.path.isfile(merge_file):
            merge_data(processed_file, merge_file)


def getNumberofRows(config_files):
    number_of_rows = [pd.read_csv(config_file).get('Cycles').count() for config_file in config_files]
    return min(number_of_rows)


def write_best_config(config_files, run_number, mode):
    config_dict = {}
    configs = {'4_40': 0, '4_60': 1, '4_80': 2, '4_100': 3, '8_40': 4, '8_60': 5, '8_80': 6, '8_100': 7}
    best_config = {'Best Configuration': []}
    for config_file in config_files:
        df = pd.read_csv(config_file)
        config = configs[config_file.split(".csv")[0].split("_{}_".format(mode))[-1]]
        phases = df['Phase'].values
        cycles = df['Cycles'].values
        for i in range(len(cycles)):
            if not phases[i] in config_dict.keys() or not config in config_dict[phases[i]].keys():
                if phases[i] in config_dict.keys():
                    config_dict[phases[i]][config] = [cycles[i]]
                else:
                    config_dict[phases[i]] = {config: [cycles[i]]}
            else:
                if config_dict[phases[i]][config] > cycles[i]:
                    config_dict[phases[i]][config] = [cycles[i]]

    for phase in config_dict.keys():
        temp = min([config_dict[phase][key][0] for key in config_dict[phase].keys()])
        for key in config_dict[phase].keys():
            if not isinstance(config_dict[phase], int) and config_dict[phase][key][0] == temp:
                config_dict[phase] = key

    for config_file in config_files:
        data = pd.read_csv(config_file, usecols=['Phase']).values
        for phase in data:
            best_config['Best Configuration'].append(config_dict[phase[0]])

    config_df = pd.DataFrame(data=best_config)
    config_df.to_csv(path_or_buf='{}_{}/best_config_file.csv'.format(mode, run_number), index=False)


def process_file(file, process_df, rows):
    integers = np.array([])
    floating = np.array([])
    cntrl = np.array([])
    cycles = np.array([])
    time_data = np.array([])
    memory = np.array([])
    logic = np.array([])
    branches = np.array([])
    jump = np.array([])
    call = np.array([])
    integers_sum = 0
    floating_sum = 0
    cntrl_sum = 0
    cycles_sum = 0
    time_avg = 0
    memory_sum = 0
    logic_sum = 0
    branches_sum = 0
    jump_sum = 0
    call_sum = 0
    feature5 = 0
    feature6 = 0
    feature7 = 0
    timeSeries = False
    for line in file:
        if timeSeries and "BlockSize = " in line:
            timeSeries = False
        if timeSeries:
            time_data = np.append(time_data, int(line.split(" ")[1]))
        if "Busy stats" in line:
            timeSeries = True
        if "[ Global ]" in line:
            Global = True
        if "[ Core 0 ]" in line:
            break
        if " = " in line:
            if "Commit.Integer" in line:
                integers = np.append(integers, int(line.split("= ")[-1].strip()))
            if "Commit.FloatingPoint" in line:
                floating = np.append(floating, int(line.split("= ")[-1].strip()))
            if "Commit.Ctrl" in line:
                cntrl = np.append(cntrl, int(line.split("= ")[-1].strip()))
            if "Cycles = " in line:
                if re.search('^Cycles = ', line):
                    cycles = np.append(cycles, int(line.split("= ")[-1].strip()))
            if "Commit.Memory" in line:
                memory = np.append(memory, int(line.split("= ")[-1].strip()))
            if "Commit.Logic" in line:
                logic = np.append(logic, int(line.split("= ")[-1].strip()))
            if "Commit.Branches" in line:
                branches = np.append(branches, int(line.split("= ")[-1].strip()))
            if "Commit.Uop.jump" in line:
                jump = np.append(jump, int(line.split("= ")[-1].strip()))
            if ("Commit.Uop.call" in line) or ("Commit.Uop.syscall" in line):
                call = np.append(call, int(line.split("= ")[-1].strip()))

    if integers.size != 0:
        integers_sum = int(np.sum(integers))
    if floating.size > 0:
        floating_sum = int(np.sum(floating))
    if cntrl.size > 0:
        cntrl_sum = int(np.sum(cntrl))
    if cycles.size > 0:
        cycles_sum = int(np.sum(cycles))
    if time_data.size > 0:
        time_avg = int(np.average(time_data))
    if memory.size > 0:
        memory_sum = int(np.sum(memory))
    if logic.size > 0:
        logic_sum = int(np.sum(logic))
    if branches.size > 0:
        branches_sum = int(np.sum(branches))
    if jump.size > 0:
        jump_sum = int(np.sum(jump))
    if call.size > 0:
        call_sum = int(np.sum(call))
    feature1 = integers_sum / 50000000000
    feature2 = floating_sum / 5000000000
    feature3 = cntrl_sum / 500000000
    feature4 = time_avg / 131072
    phase = file.name.split("\\")[-1].split("-")[-3].split("::")[-1]
    if integers_sum > 0:
        feature5 = memory_sum / (integers_sum + floating_sum + cntrl_sum + logic_sum)
        feature6 = (branches_sum - jump_sum) / jump_sum
        feature7 = call_sum / cntrl_sum
    row = [file.name.strip(), feature1, feature2, feature3, cycles_sum, feature4, feature5, feature6, feature7, phase]
    for r, i in zip(rows, row):
        process_df[r].append(i)
    return process_df


if __name__ == '__main__':
    main()
