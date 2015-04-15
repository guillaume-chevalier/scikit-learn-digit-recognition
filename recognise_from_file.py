import os
import sys
import glob

from RBM_core import *

###############################################################################
# Classification of .jpg images located in "custom_test_datas/"
# Run with "-s" arg to run in silent mode: to not show plots

show_plots = True

for arg in sys.argv:
    if arg == "-s" or arg == "-S":
        show_plots = False

print("==============================================================================")
print('           Now predicting numbers for pictures in "custom_test_datas/"')
print("==============================================================================")

path = "custom_test_datas/"
for file in glob.glob(os.path.join(path, '*.jpg')):
    print("File: \"{}\"".format(file))
    img = Image.open(file)
    local_img = convert_image_for_network(img, contrast_level=0)
    predict_2D_image(
        local_img,
        classifier,
        show_plot=show_plots
    )


print("")
print("______________________________________________________________________________")
print("Done.")
