import os
import argparse
from os.path import abspath
import cv2


def change_filename(dir, ori, new):
    origin_file = dir
    files = os.listdir(abspath(dir))
    for file in files:
        first_file = origin_file + file + "/"
        images = os.listdir(abspath(first_file))
        for image in images:
            src = first_file + image
            img = cv2.imread(src)
            if img is None:
                print("Delete file:", src)
                os.remove(src)
            split = image.split(".")
            if split is not None:
                suffix = split[-1]
                if suffix in ori:
                    src = first_file + image
                    new_image = "".join(split[:-1]) + "." + new
                    dst = first_file + new_image
                    print("Rename file:", dst)
                    os.rename(src, dst)


def is_valid(file):
    valid = True
    try:
        Image.open(file).load()
    except OSError:
        valid = False
    return valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, default="", required=True)
    parser.add_argument("--ori", "-o", type=str)
    parser.add_argument("--new", "-n", type=str, default="jpg")

    args = parser.parse_args()

    change_filename(args.path, args.ori, args.new)
    print("Change filename successful!")
