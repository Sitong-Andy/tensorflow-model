import os


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("Create new folder successful!")
    else:
        print("There is this folder!")
