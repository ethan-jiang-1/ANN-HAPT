import numpy as np
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

cdr = os.path.dirname(__file__)
if len(cdr) != 0:
    os.chdir(cdr)

from dataset_loader import load_dataset
rx_train, ry_train, rx_test, ry_test, labels, skip_ratio = load_dataset()


cpi_filename = "cp_xnn/cp_info.text"
if not os.path.isfile(cpi_filename):
    print("Error no cpi_filename {}".format(cpi_filename))
    sys.exit(-1)

with open(cpi_filename, 'r') as file:
    test_num = int(file.readline().rstrip())
    feature_num = int(file.readline().rstrip())
    skip_ratio = int(file.readline().rstrip())
    model_name = file.readline()

print("CPI", cpi_filename, test_num, feature_num, skip_ratio, model_name)


cpe_filename = "cp_study/MLPClassifierStudy.class"
if not os.path.isfile(cpe_filename):
    print("Error no cpe_filename {}".format(cpe_filename))
    sys.exit(-1)

import subprocess

root_dir = os.getcwd()
data_dir = root_dir + "/cp_xnn/dat"
working_dir = root_dir + "/cp_study"
os.chdir(working_dir)

quick_skip_ratio = 20

label_raw = []
for i in range(0, test_num, quick_skip_ratio):
    tdat = "{}/{}_{:04d}.tdat".format(data_dir, feature_num, i)
    if not os.path.isfile(tdat):
        print("failed to find prepared dat file {}, skip".format(tdat))
    else:
        with open(tdat, "r") as tdatf:
            line = tdatf.readline()
            nums = line.split(" ")

        # ret, stdout = run_command(["java", "MLPClassifier"], nums, cwd=working_dir)
        ret = True

        cmds = ["java", "MLPClassifierStudy"]
        cmds += nums 
        stdout = subprocess.check_output(cmds)
        print(i, test_num, ret, stdout, ry_test[i]-1, cmds[1])
        if ret:
            num_result = stdout.decode("utf-8").strip()
            if int(num_result) >= 0:
                label = int(num_result) + 1
                label_raw.append(label)
            else:
                print("error at {} result: {} {}".format(tdat, ret, num_result))
                label_raw.append(1)
        else:
            print("error unknown at {}".foramt(tdat))

os.chdir(root_dir)

ry_pred = np.array(label_raw)
ry_test = ry_test[::quick_skip_ratio]

from s_confusion import print_confusion_report
print_confusion_report(ry_pred, ry_test, labels)

from s_confusion import plot_confusion
plot_confusion(ry_pred, ry_test, labels, title="MLPClassifierStudy")
