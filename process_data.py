import os
import numpy as np
import csv
import re
import pandas as pd

'''
    For the given path, get the List of all files in the directory tree 
'''


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


def merge_data(processed_file, merged_file, numberOfFiles):
    df = pd.read_csv(processed_file)
    merge_dict = {}
    key_size = ((numberOfFiles/8)-1)/2
    for key in df.keys():
        merge_dict[key] = df[key].values[:int(key_size)] if key == 'Normalized time avg' else df[key].values[int(key_size)+1:]

    df_merge = pd.DataFrame(data=merge_dict)
    df_merge.to_csv(path_or_buf='./{}'.format(merged_file), index=False)


def main():
    dirName = '../blacksholes'

    # Get the list of all files in directory tree at given path
    listOfFiles = getListOfFiles(dirName)

    row = ['Filename', 'Normalized integer', 'Normalized floating', 'Normalized control', 'Cycles', 'Normalized time avg',
           'Ratio Memory', 'Ratio branches', 'Ratio call', 'Phase']
    config_files = ['processed_config_4_40.csv', 'processed_config_4_60.csv', 'processed_config_4_80.csv',
                    'processed_config_4_100.csv', 'processed_config_8_40.csv', 'processed_config_8_60.csv',
                    'processed_config_8_80.csv', 'processed_config_8_100.csv']

    merged_files = ['merged_config_4_40.csv', 'merged_config_4_60.csv', 'merged_config_4_80.csv',
                    'merged_config_4_100.csv', 'merged_config_8_40.csv', 'merged_config_8_60.csv',
                    'merged_config_8_80.csv', 'merged_config_8_100.csv']

    exists = False
    for config, merge_file in zip(config_files, merged_files):
        exists = os.path.isfile(config)
        if not exists:
            with open(config, "w", newline='') as processed_file:
                writer = csv.writer(processed_file)
                writer.writerow(row)
            processed_file.close()

    if not exists:
        for elem in listOfFiles:

            file = open(elem, "r")
            if '4core-100cache' in file.name:
                process_file(file, 'processed_config_4_100.csv')
            if '4core-80cache' in file.name:
                process_file(file, 'processed_config_4_80.csv')
            if '4core-60cache' in file.name:
                process_file(file, 'processed_config_4_60.csv')
            if '4core-40cache' in file.name:
                process_file(file, 'processed_config_4_40.csv')
            if '8core-100cache' in file.name:
                process_file(file, 'processed_config_8_100.csv')
            if '8core-80cache' in file.name:
                process_file(file, 'processed_config_8_80.csv')
            if '8core-60cache' in file.name:
                process_file(file, 'processed_config_8_60.csv')
            if '8core-40cache' in file.name:
                process_file(file, 'processed_config_8_40.csv')
            file.close()

    for processed_file, merge_file in zip(config_files, merged_files):
        if not os.path.isfile(merge_file):
            merge_data(processed_file, merge_file, len(listOfFiles))

    write_best_config(merged_files)


def write_best_config(config_files):
    df = pd.read_csv(config_files[0])
    number_of_rows = df.get('Cycles').count()
    data = np.zeros([8, number_of_rows], dtype=np.int)
    for config, j in zip(config_files, range(8)):
        df = pd.read_csv(config)
        for i in range(number_of_rows):
            data[j:j + 1, i:i + 1] = df.get('Cycles')[i]
    config_exists = os.path.isfile('best_config_file.csv')
    if not config_exists:
        with open("best_config_file.csv", "w", newline="") as best_config:
            writer = csv.writer(best_config)
            writer.writerow(['Best Configuration', 'Previous Best Configuration'])
        best_config.close()

        for i in range(number_of_rows):
            cycles_arr = np.array(
                [data[0:1, i:i + 1][0][0], data[1:2, i:i + 1][0][0], data[2:3, i:i + 1][0][0], data[3:4, i:i + 1][0][0],
                 data[4:5, i:i + 1][0][0], data[5:6, i:i + 1][0][0], data[6:7, i:i + 1][0][0],
                 data[7:8, i:i + 1][0][0]])
            sorted_cycles_arr = sorted(cycles_arr)
            with open("best_config_file.csv", "a", newline="") as best_config:
                writer = csv.writer(best_config)
                print(cycles_arr)

                print(np.where(cycles_arr == sorted(cycles_arr)[-2])[0][0])
                if sorted_cycles_arr[0] == sorted_cycles_arr[1]:
                    best_configs = [l for l, k in enumerate(cycles_arr) if k == min(cycles_arr)]
                    writer.writerow([best_configs[0], best_configs[1]])
                else:
                    writer.writerow([np.where(cycles_arr == sorted_cycles_arr[0])[0][0],
                                     np.where(cycles_arr == sorted_cycles_arr[1])[0][0]])
            best_config.close()


def process_file(file, processing_file):
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
    feature1 = 0
    feature2 = 0
    feature3 = 0
    feature4 = 0
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
    feature1 = integers_sum/500000
    feature2 = floating_sum/500000
    feature3 = cntrl_sum/500000
    feature4 = time_avg / 4096
    phase = file.name.split("\\")[-1].split("-")[-2]
    if integers_sum > 0:
        feature5 = memory_sum / (integers_sum + floating_sum + cntrl_sum + logic_sum)
        feature6 = (branches_sum - jump_sum)/jump_sum
        feature7 = call_sum / cntrl_sum
    row = [file.name.strip(), feature1, feature2, feature3, cycles_sum, feature4, feature5, feature6, feature7, phase]
    with open(processing_file, "a", newline='') as processed_file:
        writer = csv.writer(processed_file)
        writer.writerow(row)
    processed_file.close()


if __name__ == '__main__':
    main()
