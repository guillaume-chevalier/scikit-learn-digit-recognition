# Digit Recognition with scikit-learn's Bernoulli RBM and Logistic Classifier.


## DESCRIPTION

This program can read pictures from your computer's webcam to guess what number is seen. 
The number should be centred and occupying the whole webcam's field of view to be accurately recognised. 
A short time after launching the program, a window will appear to show the prediction's statistic and the image in greyscale 16x16 px that the neural network received. 
This program also can load files in a subfolder named "custom_test_datas/". 

The technology used behind this is a scikit-learn classifier containing a Bernoulli RBM (a kind of neural network), which is followed by a logistic classifier. 
To further improve this program, the classifier should learn to recognise what is not a number, it is currently forced to associate a number to the seen image. 


## REQUIREMENTS

The individual installation instructions are found in the "REQUIREMENTS" folder. 

1. Enthought Canopy (or any SciPy ecosystem)
2. scikit-learn
3. Pillow
3. openCV


## USAGE

### To plot RBM's hidden layer's weights matrices:
```sh
C:\...\scikit-learn-digit-recognition>python RBM_core.py
```
The RBM_core.py file also contains the core of the program, which is used by the next files. 


### Classification of .jpg images located in the "custom_test_datas/" folder:
```sh
C:\...\scikit-learn-digit-recognition>python recognise_from_file.py
```
Or the following, which will not show plots
```sh
C:\...\scikit-learn-digit-recognition>python recognise_from_file.py -s
```
Or simply run "custom_test_datas/classify.bat"

### Classification of images from computer's webcam:
```sh
C:\...\scikit-learn-digit-recognition>python recognise_from_webcam.py
```
Or the following, which will not show plots
```sh
C:\...\scikit-learn-digit-recognition>python  recognise_from_webcam.py -s
```


## LICENSE

BSD
