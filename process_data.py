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


def main():
    dirName = '../blacksholes'

    # Get the list of all files in directory tree at given path
    listOfFiles = getListOfFiles(dirName)

    row = ['Filename', 'Integers', 'Floating', 'Control', 'Cycles', 'Time Average']
    config_files = ['processed_config_4_100.csv', 'processed_config_4_80.csv', 'processed_config_4_60.csv',
                    'processed_config_4_40.csv', 'processed_config_8_100.csv', 'processed_config_8_80.csv',
                    'processed_config_8_60.csv', 'processed_config_8_40.csv']

    exists = False
    for config in config_files:
        exists = os.path.isfile(config)
        if not exists:
            with open(config, "w", newline='') as processed_file:
                writer = csv.writer(processed_file)
                writer.writerow(row)
            processed_file.close()

    # Print the files
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

    df = pd.read_csv(config_files[0])
    number_of_rows = df.get('Cycles').count()
    data = np.zeros([8, number_of_rows],dtype=np.int)
    for config, j in zip(config_files,range(8)):
        df = pd.read_csv(config)
        for i in range(number_of_rows):
            data[j:j + 1, i:i + 1] = df.get('Cycles')[i]

    config_exists = os.path.isfile('best_config_file.csv')
    if not config_exists:
        with open("best_config_file.csv", "w", newline="") as best_config:
            writer = csv.writer(best_config)
            writer.writerow(['Best Configuration'])
        best_config.close()

        for i in range(number_of_rows):
            cycles_arr = np.array([data[0:1, i:i+1][0][0], data[1:2, i:i+1][0][0], data[2:3, i:i+1][0][0], data[3:4, i:i+1][0][0],
                                   data[4:5, i:i + 1][0][0], data[5:6, i:i+1][0][0], data[6:7, i:i+1][0][0], data[7:8, i:i+1][0][0]])
            with open("best_config_file.csv", "a", newline="") as best_config:
                writer = csv.writer(best_config)
                writer.writerow([np.where(cycles_arr == max(cycles_arr))[0][0]])
            best_config.close()


def process_file(file, processing_file):
    integers = np.array([])
    floating = np.array([])
    cntrl = np.array([])
    cycles = np.array([])
    time_data = np.array([])
    integers_sum = 0
    floating_sum = 0
    cntrl_sum = 0
    cycles_sum = 0
    time_avg = 0
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
    row = [file.name.strip(), integers_sum, floating_sum, cntrl_sum, cycles_sum, time_avg]
    with open(processing_file, "a", newline='') as processed_file:
        writer = csv.writer(processed_file)
        writer.writerow(row)
    processed_file.close()


if __name__ == '__main__':
    main()
