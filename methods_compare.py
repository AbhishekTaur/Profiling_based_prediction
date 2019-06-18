import pandas as pd
from data_utils import getConfigFilesList


train_dict = {}
getConfigFilesList('.', False, 0, train_dict, 'train')


def test(run_number):
    df_4_40 = pd.read_csv('./train_{}/merged_config_train_4_40.csv'.format(run_number))
    df_4_60 = pd.read_csv('./train_{}/merged_config_train_4_60.csv'.format(run_number))
    df_4_80 = pd.read_csv('./train_{}/merged_config_train_4_80.csv'.format(run_number))
    df_4_100 = pd.read_csv('./train_{}/merged_config_train_4_100.csv'.format(run_number))
    df_8_40 = pd.read_csv('./train_{}/merged_config_train_8_40.csv'.format(run_number))
    df_8_60 = pd.read_csv('./train_{}/merged_config_train_8_60.csv'.format(run_number))
    df_8_80 = pd.read_csv('./train_{}/merged_config_train_8_80.csv'.format(run_number))
    df_8_100 = pd.read_csv('./train_{}/merged_config_train_8_100.csv'.format(run_number))
    best_config = pd.read_csv('./train_{}/best_config_file.csv'.format(run_number)).values
    best_config_trans = pd.read_csv('./train_{}/best_config_file_trans.csv'.format(run_number)).values
    df_keys = {0: df_4_40, 1: df_4_60, 2: df_4_80, 3: df_4_100,
               4: df_8_40, 5: df_8_60, 6: df_8_80, 7: df_8_100}

    best_config_cycles = 0
    best_config_trans_cycles = 0
    i = 0
    sim_percent = 0
    for config, config_trans in zip(best_config, best_config_trans):
        best_config_cycles = best_config_cycles + df_keys[config[0]].iloc[i, 4]
        best_config_trans_cycles = best_config_trans_cycles + df_keys[config_trans[0]].iloc[i, 4]
        if config == config_trans:
            sim_percent = sim_percent + 1

    print(best_config_cycles)
    print(best_config_trans_cycles)
        


def main():
    for key in train_dict.keys():
        test(key)

if __name__=='__main__':
    main()

