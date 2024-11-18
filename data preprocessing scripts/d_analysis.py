import os

base_dir = "C://Users//NickA//OneDrive//Documents//IIT Fall 2024//CS 512//TUM RGB-D"
x = os.listdir(base_dir)

f1 = []
f2 = []
f3 = []

for i in x:
    if i[13:22] == "freiburg1":
        f1.append(i)
    elif i[13:22] == "freiburg2":
        f2.append(i)
    elif i[13:22] == "freiburg3":
        f3.append(i)

y = os.listdir(base_dir + "//" + f1[0] + "//rgb")
print(y)