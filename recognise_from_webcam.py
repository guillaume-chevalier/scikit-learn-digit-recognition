import time
import sys

import cv2

from threading import Thread
from RBM_core import *

###############################################################################
# Classification of images from computer's webcam
# Run with "-s" arg to run in silent mode: to not show plots

show_plots = True

for arg in sys.argv:
    if arg == "-s" or arg == "-S":
        show_plots = False

process_next_image = True


def greyscale_image_and_predict(frame):
    global process_next_image
    global show_plots

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    local_img = convert_image_for_network(img)
    predict_2D_image(
        local_img,
        classifier,
        show_plot=show_plots
    )
    process_next_image = True


print("==============================================================================")
print('                 Now predicting numbers for webcam pictures')
print("==============================================================================")
print("")

cv2.namedWindow('Normal view, press ESC to exit')
try:
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
                target=greyscale_image_and_predict,
                args=([frame])
            )
            threaded_func.start()

        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            cv2.destroyWindow("Normal view, press ESC to exit")
            break
finally:
    # Free camera resources
    del vc
    threaded_func.join()

print("______________________________________________________________________________")
print("Done.")
