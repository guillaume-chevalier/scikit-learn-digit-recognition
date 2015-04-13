from RBM_core import *

###############################################################################
# Predicting from machine's vision

import time
# import sys
# import threading
# import multiprocessing
# import thread
import cv2

from threading import Thread

process_next_image = True


def convert_image_frame(local_img_frame):
    reshaped_img = grayscale_img(
        local_img_frame, inverse=True, do_contrast=True)
    reshaped_img = reshaped_img.reshape(
        [perceptron_width, reshaped_img.shape[0]/perceptron_width,
         perceptron_width, reshaped_img.shape[1]/perceptron_width]
    ).mean(3).mean(1)
    return reshaped_img


def grayscale_image_and_predict(frame):
    global process_next_image

    local_img = convert_image_frame(frame)
    prediction = predict_2D_image(
        local_img,
        classifier,
        show_plot=True
    )
    process_next_image = True


cv2.namedWindow('Normal view, press ESC to exit')
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("Normal view, press ESC to exit", frame)
    time.sleep(0.15)  # Delay
    rval, frame = vc.read()

    if (process_next_image):
        process_next_image = False
        threaded_func = Thread(
            target=grayscale_image_and_predict,
            args=([frame])
        )
        threaded_func.start()

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        #TODO: join threads and stop.
        break

# Free camera resources
#TODO: try/except this
cv2.destroyWindow("Normal view, press ESC to exit")
del vc
threaded_func.join()

print("Done.")
