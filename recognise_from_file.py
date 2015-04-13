from RBM_core import *

###############################################################################
# Predictions of images outside trained datasets

show_plots = False  # TODO: set to True.
#TODO:  currentWidth= NONE !!!! rem from "def load_grayscale... " and load dynamically.
print('"Now predicting numbers for pictures in "custom_test_datas\\".')

imgs = []
imgs.append(load_grayscale_img(path="custom_test_datas\\2.jpg", currentWidth=8, wantedWidth=perceptron_width))
imgs.append(load_grayscale_img(path="custom_test_datas\\2_2.jpg", currentWidth=16, wantedWidth=perceptron_width))
imgs.append(load_grayscale_img(path="custom_test_datas\\2_3.jpg", currentWidth=16, wantedWidth=perceptron_width))
imgs.append(load_grayscale_img(path="custom_test_datas\\3.jpg", currentWidth=16, wantedWidth=perceptron_width))
imgs.append(load_grayscale_img(path="custom_test_datas\\3_1.jpg", currentWidth=16, wantedWidth=perceptron_width))
imgs.append(load_grayscale_img(path="custom_test_datas\\3_2.jpg", currentWidth=16, wantedWidth=perceptron_width))
imgs.append(load_grayscale_img(path="custom_test_datas\\7.jpg", currentWidth=8, wantedWidth=perceptron_width))
imgs.append(load_grayscale_img(path="custom_test_datas\\7_1.jpg", currentWidth=16, wantedWidth=perceptron_width))
imgs.append(load_grayscale_img(path="custom_test_datas\\7_2.jpg", currentWidth=16, wantedWidth=perceptron_width))
imgs.append(load_grayscale_img(path="custom_test_datas\\9.jpg", currentWidth=16, wantedWidth=perceptron_width))
# imgs.append(X_test[1001])
# imgs.append(X_test[2020])
# imgs.append(X_test[3004])
# imgs.append(X_test[4005])
# imgs.append(X_test[5009])

for img in imgs:
    predicted_num = predict_2D_image(
        img,
        classifier,
        show_plot=show_plots
    )
    # print(np.argmax(classifier.predict_log_proba([img.flatten()])))

print("Done.")
