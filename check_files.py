import os

benchmark_files = {'blackscholes': [], 'dedup': [], 'streamcluster': [], 'swaptions': [], 'freqmine': [],
                   'fluidanimate': [], 'canneal': []}


def getFiles(benchmark, dirName):
    for root, dirs, files in os.walk(dirName):
        if len(dirs) == 0 and benchmark in root:
            benchmark_files[benchmark] = benchmark_files[benchmark] + [os.path.join(root, file) for file in files]
    benchmark_files[benchmark] = sorted(benchmark_files[benchmark])


def main():
    for benchmark in benchmark_files.keys():
        getFiles(benchmark, '../data')
        file = open(benchmark + '.txt', 'w')
        file.write(str(benchmark_files[benchmark]))


if __name__=='__main__':
    main()
