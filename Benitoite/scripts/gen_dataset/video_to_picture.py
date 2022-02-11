import cv2
import time
from mkdir import mkdir


def video2pic(video_path):
    vc = cv2.VideoCapture(video_path)
    c = 0
    rval = vc.isOpened()
    timeF = 30
    out_path = (
        "../../dataset/"
        + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        + "/"
    )
    mkdir(out_path)
    idx = 1
    while rval:
        c += 1
        rval, frame = vc.read()
        if c % timeF == 0:
            file_name = out_path + str(idx).zfill(5) + ".jpg"
            cv2.imwrite(file_name, frame)
            idx += 1
            print("Generate picture ", file_name)
        cv2.waitKey(1)
    print("Grab pictures successful!")
    vc.release()


video2pic("/Users/sitongfeng/Downloads/IMG_2812.MOV")
