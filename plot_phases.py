import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

merged_files = ["merged_config_train_4_40.csv", "merged_config_train_4_60.csv", "merged_config_train_4_80.csv",
                "merged_config_train_4_100.csv",
                "merged_config_train_8_40.csv", "merged_config_train_8_60.csv", "merged_config_train_8_80.csv",
                "merged_config_train_8_100.csv",]


def plot_cycles(df, cores, cache):
    plt.figure(figsize=(25, 15))
    x = np.arange(len(df.get("Cycles")))
    plt.bar(x, height=df.get("Cycles"), align='center')
    plt.xlabel("Time interval")
    plt.ylabel("Cycles")
    plt.title("Cycles versus time interval")
    plt.savefig('Cycles_{}_{}.png'.format(cores, cache))


def plot_phases(df, cores, cache):
    plt.figure(figsize=(25, 15))
    x = np.arange(len(df.get("Phase")))
    plt.bar(x, height=df.get("Phase"), align='center')
    plt.xlabel("Time interval")
    plt.ylabel("Phase")
    plt.title("Phase versus time interval")
    plt.savefig('Phase_{}_{}.png'.format(cores, cache))


def main():
    for merged_file in merged_files:
        df = pd.read_csv("train_4050/{}".format(merged_file))
        plot_cycles(df, merged_file.split("_")[-2], merged_file.split("_")[-1])
        plot_phases(df, merged_file.split("_")[-2], merged_file.split("_")[-1])


if __name__=="__main__":
    main()

