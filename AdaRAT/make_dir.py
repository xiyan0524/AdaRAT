import os

name = "result"
dataset = ["cifar10", "cifar100"]
dir = ["data", "save_model", "txt"]

if not os.path.exists(name):
    os.makedirs(name)
    print(str(name) + ": Folder created!")
else:
    print(str(name) + ": Folder already exists!")

for i in range(len(dataset)):
    path = name + '/' + dataset[i]
    if not os.path.exists(path):
        os.makedirs(path)
        print(str(path) + ": Folder created!")
    else:
        print(str(path) + ": Folder already exists!")

    for j in range(len(dir)):
        path = name + '/' + dataset[i] + '/' + dir[j]
        if not os.path.exists(path):
            os.makedirs(path)
            print(str(path) + ": Folder created!")
        else:
            print(str(path) + ": Folder already exists!")
