# This work is much pimped from this example of scikit-learn's documentation:
# http://scikit-learn.org/stable/auto_examples/neural_networks/plot_rbm_logistic_classification.html
"""
==============================================================
Restricted Boltzmann Machine features for digit classification
==============================================================

For grayscale image data where pixel values can be interpreted as degrees of
blackness on a white background, like handwritten digit recognition, the
Bernoulli Restricted Boltzmann machine model can perform effective non-linear
feature extraction.

In order to learn good latent representations from a small dataset, we
artificially generate more labeled data by perturbing the training data with
linear shifts of 1 pixel in each direction. The same processing will be done
when guessing an image's label.

There is a classification pipeline with a BernoulliRBM
feature extractor and a LogisticRegression classifier. The hyperparameters
of the entire model (learning rate, hidden layer size, regularization)
were optimized by grid search (cross validation), but the search has not been
done with a big amount of parameters.
"""

from __future__ import print_function

print(__doc__)

# Authors: Yann N. Dauphin, Vlad Niculae, Gabriel Synnaeve, Guillaume Chevalier
# License: BSD

import pickle
import os.path
import math
import random
import Image

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

from scipy.ndimage import convolve
from sklearn import datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression

###############################################################################
# Defining functions & misc.

perceptron_width = 16  # Should be a multiple of 8. If changed, also edit the line at # TODO: replace "2"
perceptron_count = perceptron_width * perceptron_width
pickles_suffix = "_{0}x{0}.pickle".format(perceptron_width)

hidden_layer_width = 32
hidden_layer_count = hidden_layer_width * hidden_layer_width


def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    and keeping an original version of the images.
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((perceptron_width, perceptron_width)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y


def shuffle_dataset(X, Y):
    """
    Shuffle two lists keeping their elements
    associated: Xi == Yi.
    """
    z = zip(X, Y)
    random.shuffle(z)
    return zip(*z)


def resample_img(img, currentWidth=128, newWidth=perceptron_width):
    """
    Changes the size of an image.
    Yet, if the image needs to be scaled up,
    the image will only enlarge by a factor of 2 by 2.
    """
    inv_ratio = currentWidth/newWidth
    if (inv_ratio < 1):
        return scipy.ndimage.zoom(img, 2, order=1)  # TODO: replace "2" by int(1/inv_ratio)
    else:
        return img.reshape([newWidth, inv_ratio, newWidth, inv_ratio]).mean(3).mean(1)


def scale_elements_values_from_0_to_1(arr):
    """
    Scale any multidimensional array's items' values
    from 0 to 1, keeping relative proportions.
    Each item needs to be a number.
    """
    amax = np.amax(arr)
    amin = np.amin(arr)
    arr = np.subtract(arr, amin)
    P = 1/(amax-amin)
    arr = P*np.array(arr)
    return arr


def rectify(img, strong_rectification=True):
    """
    Rectify from 0 to 1 each element of
    any multidimensional array.

    If strong_rectification is set to true,
    values will be rectified from 0.35 to 0.7,
    and values equal to that will then be
    remapped to 0 and 1.
    """
    if (strong_rectification):
        def rectify_f(pixel):
            if (pixel <= 0.35):
                pixel = 0.0
            elif (pixel >= 0.7):
                pixel = 1.0
            return pixel
    else:
        def rectify_f(pixel):
            if (pixel < 0.0):
                pixel = 0.0
            elif (pixel > 1.0):
                pixel = 1.0
            return pixel
    rectify_f = np.vectorize(rectify_f)
    return rectify_f(img)


def contrast(img, do_rectify=True):
    """
    Apply a custom contrast filter to any multidimensional
    array which elements' values range from 0 to 1.
    """
    def contrast_f(pixel):
        # Those values are set to fit my webcam
        pixel = 1.4/math.pi*math.atan(4*(pixel-0.49))+0.5
        if (pixel < 0.35):
            pixel = 0.0
        elif (pixel > 0.7):
            pixel = 1.0
        return pixel
    contrast_f = np.vectorize(contrast_f)
    return rectify(contrast_f(img), do_rectify)


def grayscale_img(color_image, inverse=True, do_contrast=False):
    """
    Convert an RGB image (3D array) to a
    grayscale 2D array, which elements range
    from 0 to 1.
    """
    img = np.array(color_image)

    if (inverse):
        def iavg(a):
            return (255.0 - (np.average(a)))
    else:
        def iavg(a):
            return (np.average(a))

    grayscale = np.zeros((img.shape[0], img.shape[1]))
    for rownum in range(len(img)):
        for colnum in range(len(img[rownum])):
            grayscale[rownum][colnum] = iavg(img[rownum][colnum])
    grayscale = scale_elements_values_from_0_to_1(grayscale)

    if (do_contrast):
        grayscale = contrast(grayscale)

    return grayscale


def load_grayscale_img(path="7.jpg", currentWidth=8, wantedWidth=perceptron_width):
    """
    Load an image to a grayscale 2D array with
    elements' values ranging from 0 to 1.
    """
    im = Image.open(path)
    return resample_img(grayscale_img(im), currentWidth, newWidth=wantedWidth)



def predict_2D_image(img, classifier, show_plot=False):
    """
    Predict the number associated to an imae from a trained
    trained scikit-learn classifier.
    Optionnaly shows the result of the prediction in a plot.
    """
    predicted_num = classifier.predict(img.flatten())[0]
    print("Predicted number: {}".format(predicted_num))

    if (show_plot):
        bars_width = 0.8
        #TODO: which one?
        decision_function_vals = classifier.decision_function(img.flatten())[0]
        # decision_function_vals = classifier.predict_log_proba(img.flatten())[0]

        confidence_labels = range(len(decision_function_vals))
        confidence_labels_pos = np.arange(bars_width/2, len(decision_function_vals)+bars_width/2, 1)

        plt.subplots_adjust(hspace=0.5)

        ax1 = plt.subplot(2, 1, 1)
        ax1.imshow(img, cmap=plt.cm.gray_r,
                   interpolation='nearest')
        ax1.set_title("Predicted value: {}".format(predicted_num), fontsize=22)

        ax2 = plt.subplot(2, 1, 2)
        ax2.bar(confidence_labels, decision_function_vals, bars_width)
        # plt.suptitle('Confidence decision function for each label')
        ax2.set_title('Confidence decision function for each label', fontsize=22)
        plt.ylabel('Confidence')
        plt.xlabel('Label')
        ax2.plot([0, 10], [0, 0], 'k-', lw=2)

        ax2.set_xticks(confidence_labels_pos)
        ax2.set_xticklabels(confidence_labels)

        plt.show()
    return predicted_num


###############################################################################
# Load Data

datasets_path = 'dataset_pickles\\datasets{}'.format(pickles_suffix)
if (os.path.exists(datasets_path)):
    with open(datasets_path) as f:
        X, Y = pickle.load(f)

else:
    def load_default_dataset():
        """
        Fancy "datasets.load_digits()". Returns a list
        of flat arrays (lists) representing grayscale images, and
        their associated labels.
        """
        images = []
        labels = []

        pickle_path = "dataset_pickles\dflt_dataset{}".format(pickles_suffix)
        if (os.path.exists(pickle_path)):
            with open(pickle_path) as f:
                images, labels = pickle.load(f)
        else:
            digits_set = datasets.load_digits()
            original_images = np.asarray(digits_set.data, 'float32')

            if(perceptron_width != 8):  # Resampling if needed
                xi = 0
                for x in original_images:
                    this_image = resample_img(x.reshape(8, 8), currentWidth=8, newWidth=perceptron_width).flatten()
                    images.append(this_image)
                    xi += 1
            images = (images - np.min(images, 0)) / (np.max(images, 0) + 0.0001)  # 0-1 scaling
            labels = digits_set.target

            with open(pickle_path, 'w') as f:
                pickle.dump([images, labels], f)
                print()

        return images, labels

    def load_fnt_dataset():
        """
        Loads the Chars74K's "Fnt" dataset. Returns a list
        of flat arrays (lists) representing grayscale images, and
        their associated labels.
        Can be downloaded from: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
        Direct download link: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz
        TODO: CITE in http://scholar.google.co.uk/citations?hl=en&user=RzL_a1gAAAAJ&view_op=list_works
        """
        images = []
        labels = []

        for i in xrange(0+1, 1016+1):
            for x in xrange(0+1, 9+1+1):
                folder_name = "Sample{0:03d}".format(x)
                file_name_prefix = "img{0:03d}-".format(x)

                file_name = "{0}{1:05d}.png".format(file_name_prefix, i)
                img_path = "fnt_dataset\{}\{}".format(folder_name, file_name)

                pickle_file_name = "{}{}".format(file_name[0:-4], pickles_suffix)
                pickle_path = "dataset_pickles\{}".format(pickle_file_name)

                if (os.path.exists(pickle_path)):
                    with open(pickle_path) as f:
                        this_data_img, this_target_label = pickle.load(f)
                else:
                    this_data_img = load_grayscale_img(path=img_path, currentWidth=128, wantedWidth=perceptron_width).flatten()
                    this_target_label = x-1
                    with open(pickle_path, 'w') as f:
                        pickle.dump([this_data_img, this_target_label], f)

                images.append(this_data_img)
                labels.append(this_target_label)
        return images, labels

    print("Loading default dataset...")
    x_dflt, y_dflt = load_default_dataset()
    print("Loading fnt dataset...")
    x_fnt, y_fnt = load_fnt_dataset()

    print("Appending datasets...")
    X, Y = np.append(x_dflt, x_fnt, axis=0), np.append(y_dflt, y_fnt, axis=0)

    # Saving the objects:
    with open(datasets_path, 'w') as f:
        pickle.dump([X, Y], f)

initial_test_size_split = 0.15
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=initial_test_size_split,
                                                    random_state=42)

# Increase and noise training data, but not testing data
X_train, Y_train = nudge_dataset(X_train, Y_train)
# This redefines the test data ratio as:
test_size = len(X_test) * 1. / len(X_train)

print("Setup Done:")
print("Total images: {}".format(
    len(X)*(initial_test_size_split+5*(1-initial_test_size_split))
))
print("Total training images: {}".format(len(X_train)))
print("Total testing images: {}".format(len(X_test)))
print("Testing size: {}%".format(test_size*100))
print("")


###############################################################################
# Cross validation (CV) to find hyper-parameters

# The Cross Validation technique used is inspired from:
# http://www.pyimagesearch.com/2014/06/23/applying-deep-learning-rbm-mnist-using-python/

# If a CV has already been done with the actual "hidden_layer_count" variable.
CV_already_done = True

if (not CV_already_done):
    # initialize the RBM + Logistic Regression pipeline
    rbm = BernoulliRBM(random_state=1, verbose=True)
    logistic = LogisticRegression()
    classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

    # perform a grid search on the learning rate, number of
    # iterations, and number of components on the RBM and
    # C for Logistic Regression
    print("SEARCHING RBM AND LOGISTIC REGRESSION'S HYPER-PARAMETERS")
    # The following params already have been edited manually
    # to then wiggle between these remaining values.
    params = {
        'rbm__learning_rate': [0.005, 0.007, 0.008],
        'rbm__n_iter': [45],
        'rbm__n_components': [hidden_layer_count],
        'logistic__C': [1000.0]}

    # perform a grid search over the parameter
    gs = GridSearchCV(classifier, params, n_jobs=-1, verbose=1, cv=2)
    gs.fit(X_train, Y_train)

    # print diagnostic information to the user and grab the
    # best model
    print("best score: %0.3f" % (gs.best_score_))
    print("RBM + LOGISTIC REGRESSION'S HYPER-PARAMETERS")

    CV_result = gs.best_estimator_.get_params()

else:
    # CV Results from a hidden_layer_count = 15.
    CV_result = {
        'logistic__penalty': 'l2',
        'rbm__verbose': True,
        'logistic__tol': 0.0001,
        'logistic__dual': False,
        'logistic__fit_intercept': True,
        'rbm': BernoulliRBM(batch_size=10, learning_rate=0.007,
                            n_components=hidden_layer_count, n_iter=45,
                            random_state=1, verbose=True),
        'rbm__n_iter': 45,
        'rbm__learning_rate': 0.007,
        'logistic__class_weight': None,
        'logistic': LogisticRegression(C=1000.0, class_weight=None, dual=False,
                                       fit_intercept=True, intercept_scaling=1,
                                       penalty='l2', random_state=None, tol=0.0001),
        'rbm__n_components': hidden_layer_count,
        'logistic__C': 1000.0,
        'logistic__random_state': None,
        'rbm__batch_size': 10,
        'rbm__random_state': 1,
        'logistic__intercept_scaling': 1
    }

print("Neural network's parameters:")
print(CV_result["rbm"])
print(CV_result["logistic"])
# print(CV_result)


###############################################################################
# Training from CV result parameters and evaluation.

RBM_classifier_path = 'RBM_classifier_{}_{}{}'.format(
    hidden_layer_count, CV_result["rbm"].n_iter, pickles_suffix)

if (os.path.exists(RBM_classifier_path)):
    with open(RBM_classifier_path) as f:
        rbm, logistic, classifier, metrics_results = pickle.load(f)
else:
    rbm = CV_result["rbm"]
    logistic = CV_result["logistic"]
    classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

    # Training RBM-Logistic Pipeline
    classifier.fit(X_train, Y_train)

    # Evaluation
    metrics_results = metrics.classification_report(
        Y_test,
        classifier.predict(X_test))

    # Saving the objects:
    with open(RBM_classifier_path, 'w') as f:
        pickle.dump([rbm, logistic, classifier, metrics_results], f)

print()
print("Test results of logistic regression using RBM features:\n{}\n".format(
    metrics_results)
)


if __name__ == "__main__":
    ###############################################################################
    # Plotting 1st RBM hidden layer's weight matrix

    print("Plotting 1st RBM hidden layer's weight matrix.")
    print("Preparing plot", end="")
    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(rbm.components_):
        if (i % 50 == 0):
            print(".", end="")
        plt.subplot(hidden_layer_width, hidden_layer_width, i + 1)
        plt.imshow(comp.reshape((perceptron_width, perceptron_width)),
                   cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle("RBM's {} hidden layer's weights matrixes".format(
        hidden_layer_count), fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    print(" Done.")

    plt.show()
