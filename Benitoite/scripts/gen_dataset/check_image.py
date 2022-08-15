#!/usr/bin/env python3
from genericpath import isfile
import os
import argparse
from os.path import abspath
import tensorflow as tf


def is_valid_type(file):
    invalid_type = ["gif", "jpeg", "png", "bmp"]
    valid = True
    try:
        if file.split(".")[-1].lower() in invalid_type:
            valid = False
    except OSError:
        valid = False
    return valid


def validate_image(dir):
    images = os.listdir(abspath(dir))
    invalid_image = []
    for i in images:
        src = dir + i
        image = tf.io.read_file(src)
        try:
            image = tf.io.decode_image(image, channels=3)
        except Exception as e:
            img_endwith = ["jpg", "png", "bmp", "gif", "jpeg"]
            is_image = (
                True
                if "." in src and src.split(".")[-1].lower() in img_endwith
                else False
            )
            if is_image:
                print(i, " ", e)
                invalid_image.append(src)
    print("Checking finished, invalid image number:", len(invalid_image))
    return invalid_image


def check_image_valid(dir):
    origin_file = dir
    invalid_type = []
    images = os.listdir(abspath(dir))
    for image in images:
        src = origin_file + image
        if not is_valid_type(src):
            invalid_type.append(image)
            print("Invalid image type: ", src)
    print(
        "Checking finished, invalid image number:",
        len(invalid_type),
    )
    return invalid_type


def modify_image(image_path, modify_type):
    img_endwith = ["jpg", "png", "bmp", "gif", "jpeg"]
    for path in image_path:
        is_image = (
            True
            if "." in path and path.split(".")[-1].lower() in img_endwith
            else False
        )
        if is_image and modify_type == "remove":
            os.remove(path)
            print("Removed: ", path)
        elif is_image and modify_type == "rename":
            new_image_path = path.split("/")[-1] + ".jpg"
            os.rename(path, new_image_path)
            print("Renamed image from ", path, " to ", new_image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", "-p", type=str, default="./", help="Path to the directory"
    )

    args = parser.parse_args()
    # FIRST CHECK THE INVALID IMAGE
    invalid_image_type = check_image_valid(args.path)
    if invalid_image_type:
        modify_image(invalid_image_type, "rename")
    # SECOND CHECK THE INVALID IMAGE
    invalid_image_tf = validate_image(args.path)
    if invalid_image_tf:
        modify_image(invalid_image_tf, "remove")

    print("Repair dataset complete!")
