import os
import datetime
from random import sample
from random import randint
import threading
import logging
import pandas as pd
from benchmark import Benchmark
import re

program_dict = {0: 'blackscholes', 1: 'dedup', 2: 'streamcluster', 3: 'swaptions', 4: 'freqmine', 5: 'fluidanimate',
                6: 'canneal'}

benchmark_files = {'blackscholes': [], 'dedup': [], 'streamcluster': [], 'swaptions': [], 'freqmine': [],
                   'fluidanimate': [], 'canneal': []}

process_row = ['Filename', 'Normalized integer', 'Normalized floating', 'Normalized control', 'Cycles',
               'Normalized time avg', 'Ratio Memory', 'Ratio branches', 'Ratio call', 'Phase']

config_files = ['4core-100', '4core-80', '4core-60', '4core-40', '8core-100', '8core-80', '8core-60', '8core-40']


def return_dict():
    return {'Filename': [], 'Normalized integer': [], 'Normalized floating': [], 'Normalized control': [], 'Cycles': [],
            'Normalized time avg': [],
            'Ratio Memory': [], 'Ratio branches': [], 'Ratio call': [], 'Phase': []}


def getFiles(benchmark, dirName):
    for root, dirs, files in os.walk(dirName):
        if len(dirs) == 0 and benchmark in root:
            benchmark_files[benchmark] = benchmark_files[benchmark] + [os.path.join(root, file) for file in files]
    benchmark_files[benchmark] = sorted(benchmark_files[benchmark])


def getLeastNumberOfRows(merge_dict):
    min_rows = min([len(merge_dict[key]) for key in merge_dict.keys()])
    for key in merge_dict.keys():
        merge_dict[key] = merge_dict[key][0:min_rows]


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
    print("Starting the program")

    # Get the list of all files in directory tree at given path

    subset = sample([i for i in range(6)], 5)
    run_number = str(randint(0, 10000))
    print(subset)
    train_subset = []
    test_subset = []
    for benchmark in benchmark_files.keys():
        getFiles(benchmark, '../data')

    for key in subset:
        train_subset.append(program_dict[key])
    for key in range(7):
        if key not in subset:
            test_subset.append(program_dict[key])

    print(test_subset)
    print(train_subset)
    train_data_folder = 'train_{}'.format(run_number)
    test_data_folder = 'test_{}'.format(run_number)
    benchmark_folder = 'benchmark_files'
    if not os.path.exists(train_data_folder):
        os.mkdir(test_data_folder)
        os.mkdir(train_data_folder)
        print("Created directories: {}".format(train_data_folder))
        print("Created directories: {}".format(test_data_folder))
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    blackscholes = Benchmark('blackscholes', benchmark_files['blackscholes'], process_row, return_dict())
    dedup = Benchmark('dedup', benchmark_files['dedup'], process_row, return_dict())
    streamcluster = Benchmark('streamcluster', benchmark_files['streamcluster'], process_row, return_dict())
    swaptions = Benchmark('swaptions', benchmark_files['swaptions'], process_row, return_dict())
    freqmine = Benchmark('freqmine', benchmark_files['freqmine'], process_row, return_dict())
    fluidanimate = Benchmark('fluidanimate', benchmark_files['fluidanimate'], process_row, return_dict())
    canneal = Benchmark('canneal', benchmark_files['canneal'], process_row, return_dict())

    benchmarks = [blackscholes, dedup, streamcluster, swaptions, freqmine, fluidanimate, canneal]
    threads = list()

    if not os.path.exists(benchmark_folder):
        os.mkdir(benchmark_folder)
        for benchmark in benchmarks:
            x = threading.Thread(target=benchmark.process_files())
            threads.append(x)
            x.start()

        for index, thread in enumerate(threads):
            logging.info("Main    : before joining thread %d.", index)
            thread.join()
            logging.info("Main    : thread %d done", index)

        for benchmark, benchmark_name in zip(benchmarks, benchmark_files.keys()):
            if benchmark_name in train_subset:
                pd.DataFrame(benchmark.process_dict).to_csv(
                    path_or_buf='{}/{}.csv'.format(benchmark_folder, benchmark_name), index=False)
            if benchmark_name in test_subset:
                pd.DataFrame(benchmark.process_dict).to_csv(
                    path_or_buf='{}/{}.csv'.format(benchmark_folder, benchmark_name), index=False)
    else:
        for benchmark, benchmark_name in zip(benchmarks, benchmark_files.keys()):
            benchmark.process_dict = pd.read_csv('{}/{}.csv'.format(benchmark_folder, benchmark_name)).to_dict()

    process_df_4_100_train = pd.DataFrame(data=return_dict())
    process_df_4_80_train = pd.DataFrame(data=return_dict())
    process_df_4_60_train = pd.DataFrame(data=return_dict())
    process_df_4_40_train = pd.DataFrame(data=return_dict())
    process_df_8_100_train = pd.DataFrame(data=return_dict())
    process_df_8_80_train = pd.DataFrame(data=return_dict())
    process_df_8_60_train = pd.DataFrame(data=return_dict())
    process_df_8_40_train = pd.DataFrame(data=return_dict())
    process_df_4_100_test = pd.DataFrame(data=return_dict())
    process_df_4_80_test = pd.DataFrame(data=return_dict())
    process_df_4_60_test = pd.DataFrame(data=return_dict())
    process_df_4_40_test = pd.DataFrame(data=return_dict())
    process_df_8_100_test = pd.DataFrame(data=return_dict())
    process_df_8_80_test = pd.DataFrame(data=return_dict())
    process_df_8_60_test = pd.DataFrame(data=return_dict())
    process_df_8_40_test = pd.DataFrame(data=return_dict())

    for config in config_files:
        for benchmark, benchmark_name in zip(benchmarks, benchmark_files.keys()):
            df = pd.DataFrame(benchmark.process_dict)
            if config == '4core-100':
                if len(df):
                    if benchmark_name in train_subset:
                        process_df_4_100_train = process_df_4_100_train.append(df[df['Filename'].str.contains(config)])
                    if benchmark_name in test_subset:
                        process_df_4_100_test = process_df_4_100_test.append(df[df['Filename'].str.contains(config)])
            if config == '4core-80':
                if len(df):
                    if benchmark_name in train_subset:
                        process_df_4_80_train = process_df_4_80_train.append(df[df['Filename'].str.contains(config)])
                    if benchmark_name in test_subset:
                        process_df_4_80_test = process_df_4_80_test.append(df[df['Filename'].str.contains(config)])
            if config == '4core-60':
                if len(df):
                    if benchmark_name in train_subset:
                        process_df_4_60_train = process_df_4_60_train.append(df[df['Filename'].str.contains(config)])
                    if benchmark_name in test_subset:
                        process_df_4_60_test = process_df_4_60_test.append(df[df['Filename'].str.contains(config)])
            if config == '4core-40':
                if len(df):
                    if benchmark_name in train_subset:
                        process_df_4_40_train = process_df_4_40_train.append(df[df['Filename'].str.contains(config)])
                    if benchmark_name in test_subset:
                        process_df_4_40_test = process_df_4_40_test.append(df[df['Filename'].str.contains(config)])
            if config == '8core-100':
                if len(df):
                    if benchmark_name in train_subset:
                        process_df_8_100_train = process_df_8_100_train.append(df[df['Filename'].str.contains(config)])
                    if benchmark_name in test_subset:
                        process_df_8_100_test = process_df_8_100_test.append(df[df['Filename'].str.contains(config)])
            if config == '8core-80':
                if len(df):
                    if benchmark_name in train_subset:
                        process_df_8_80_train = process_df_8_80_train.append(df[df['Filename'].str.contains(config)])
                    if benchmark_name in test_subset:
                        process_df_8_80_test = process_df_8_80_test.append(df[df['Filename'].str.contains(config)])
            if config == '8core-60':
                if len(df):
                    if benchmark_name in train_subset:
                        process_df_8_60_train = process_df_8_60_train.append(df[df['Filename'].str.contains(config)])
                    if benchmark_name in test_subset:
                        process_df_8_60_test = process_df_8_60_test.append(df[df['Filename'].str.contains(config)])
            if config == '8core-40':
                if len(df):
                    if benchmark_name in train_subset:
                        process_df_8_40_train = process_df_8_40_train.append(df[df['Filename'].str.contains(config)])
                    if benchmark_name in test_subset:
                        process_df_8_40_test = process_df_8_40_test.append(df[df['Filename'].str.contains(config)])

    process_df_train = {'{}/processed_config_{}_4_100.csv'.format(train_data_folder, 'train'): process_df_4_100_train,
                        '{}/processed_config_{}_4_80.csv'.format(train_data_folder, 'train'): process_df_4_80_train,
                        '{}/processed_config_{}_4_60.csv'.format(train_data_folder, 'train'): process_df_4_60_train,
                        '{}/processed_config_{}_4_40.csv'.format(train_data_folder, 'train'): process_df_4_40_train,
                        '{}/processed_config_{}_8_100.csv'.format(train_data_folder, 'train'): process_df_8_100_train,
                        '{}/processed_config_{}_8_80.csv'.format(train_data_folder, 'train'): process_df_8_80_train,
                        '{}/processed_config_{}_8_60.csv'.format(train_data_folder, 'train'): process_df_8_60_train,
                        '{}/processed_config_{}_8_40.csv'.format(train_data_folder, 'train'): process_df_8_40_train}
    process_df_test = {'{}/processed_config_{}_4_100.csv'.format(test_data_folder, 'test'): process_df_4_100_test,
                       '{}/processed_config_{}_4_80.csv'.format(test_data_folder, 'test'): process_df_4_80_test,
                       '{}/processed_config_{}_4_60.csv'.format(test_data_folder, 'test'): process_df_4_60_test,
                       '{}/processed_config_{}_4_40.csv'.format(test_data_folder, 'test'): process_df_4_40_test,
                       '{}/processed_config_{}_8_100.csv'.format(test_data_folder, 'test'): process_df_8_100_test,
                       '{}/processed_config_{}_8_80.csv'.format(test_data_folder, 'test'): process_df_8_80_test,
                       '{}/processed_config_{}_8_60.csv'.format(test_data_folder, 'test'): process_df_8_60_test,
                       '{}/processed_config_{}_8_40.csv'.format(test_data_folder, 'test'): process_df_8_40_test}

    for processed_file in process_df_train.keys():
        process_df_train[processed_file].to_csv(path_or_buf=processed_file,
                                                index=False)
    for processed_file in process_df_test.keys():
        process_df_test[processed_file].to_csv(path_or_buf=processed_file,
                                               index=False)
    merged_train_files = ['{}/merged_config_{}_4_100.csv'.format(train_data_folder, 'train'),
                          '{}/merged_config_{}_4_80.csv'.format(train_data_folder, 'train'),
                          '{}/merged_config_{}_4_60.csv'.format(train_data_folder, 'train'),
                          '{}/merged_config_{}_4_40.csv'.format(train_data_folder, 'train'),
                          '{}/merged_config_{}_8_100.csv'.format(train_data_folder, 'train'),
                          '{}/merged_config_{}_8_80.csv'.format(train_data_folder, 'train'),
                          '{}/merged_config_{}_8_60.csv'.format(train_data_folder, 'train'),
                          '{}/merged_config_{}_8_40.csv'.format(train_data_folder, 'train')]
    merged_test_files = ['{}/merged_config_{}_4_100.csv'.format(test_data_folder, 'test'),
                         '{}/merged_config_{}_4_80.csv'.format(test_data_folder, 'test'),
                         '{}/merged_config_{}_4_60.csv'.format(test_data_folder, 'test'),
                         '{}/merged_config_{}_4_40.csv'.format(test_data_folder, 'test'),
                         '{}/merged_config_{}_8_100.csv'.format(test_data_folder, 'test'),
                         '{}/merged_config_{}_8_80.csv'.format(test_data_folder, 'test'),
                         '{}/merged_config_{}_8_60.csv'.format(test_data_folder, 'test'),
                         '{}/merged_config_{}_8_40.csv'.format(test_data_folder, 'test')]

    for processed_file, merge_file in zip(process_df_train.keys(), merged_train_files):
        if not os.path.isfile(merge_file):
            merge_data(processed_file, merge_file)
    for processed_file, merge_file in zip(process_df_test.keys(), merged_test_files):
        if not os.path.isfile(merge_file):
            merge_data(processed_file, merge_file)

    write_best_config(merged_train_files, run_number, 'train')
    write_best_config(merged_test_files, run_number, 'test')
    end = datetime.datetime.now()
    print("End time: ", end)
    total_time = end - start
    print("Total time in seconds: ", int(total_time.total_seconds()))


if __name__ == '__main__':
    main()
