import numpy as np
import re


class Benchmark:
    def __init__(self, benchmark, listOfFiles, rows, process_dict):
        self.benchmark = benchmark
        self.listOfFiles = listOfFiles
        self.rows = rows
        self.process_dict =process_dict

    def process_files(self):
        for elem in self.listOfFiles:
            file = open(elem, 'r')
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
            row = [file.name.strip(), feature1, feature2, feature3, cycles_sum, feature4, feature5, feature6, feature7,
                   phase]
            for r, i in zip(self.rows, row):
                self.process_dict[r].append(i)