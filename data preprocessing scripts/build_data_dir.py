import os
import random

def split(num_list):
    test = random.sample(num_list, int(len(num_list) * 0.2))
    train_all = [i for i in num_list if i not in test]
    val = random.sample(train_all, int(len(train_all) * 0.2))
    train = [i for i in train_all if i not in val]

    return train, val, test


def make_file(file_name, nums):
    f = open(file_name, "w")
    for i in nums:
        f.write("scan " + str(i) + "\n")

def build_dirs(base_dir, n):
    os.mkdir(base_dir)
    for i in range(n):
        os.makedirs(base_dir + "/scan " + str(i) + "/color")
        os.mkdir(base_dir + "/scan " + str(i) + "/depth")


def make_all(base_dir, n):
    build_dirs(base_dir, n)
    train, val, test = split([i for i in range(n)])
    make_file(base_dir + "/train.txt", train)
    make_file(base_dir + "/val.txt", val)
    make_file(base_dir + "/test.txt", test)

make_all("data", 11)