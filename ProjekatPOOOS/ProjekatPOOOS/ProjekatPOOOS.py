import matplotlib.pyplot  as plt
import csv
import matplotlib
import random
import sys
import cv2
import numpy as np
from scipy.ndimage.filters import median_filter
import glob
import PIL
import pandas as pd  
import argparse
import imutils
import os
from imutils import paths
from PIL import Image
from resizeimage import resizeimage
from skimage.feature import hog
from skimage import data, exposure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
from sklearn.externals import joblib

#UCITAVA SLIKE SA DATA SETA

def readTrafficSigns(rootpath = r"E:\ajdin\4. godina\POOS\traffic_sign_recognition_poos-master"):
    images = [] # images
    labels = [] # corresponding labels
    koordinate = []
    # loop over all 42 classes
    for c in range(0,3):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            x1 = row[3]
            y1 = row[4]
            x2 = row[5]
            y2 = row[6]
            info = x2, y2, x1, y1
            koordinate.append(info)     
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return images, labels, koordinate

#KROPA SLIKE 
def krop(koordinate, slike):
    kropano = []
    for i in range(len(slike)):
        x2, y2, x1, y1 = koordinate[i]
        img = slike[i]
        crop_img = img[int(y1):int(y2), int(x1):int(x2)]
        kropano.append(crop_img)
    return kropano

#SPREMANJE SLIKE
def spremi(slike, putanja):
    for i in range(len(slike)):
        plt.imsave(putanja + str(i), slike[i]) 


#UKLANJA SUM
def ukloniSum(slika):
    return cv2.fastNlMeansDenoisingColored(slika,None,10,10,7,21)

#UKLANJA NEOSTRINE
def ukloniNeostrinu(slike, k=0.7):
    bezNeostrina = []
    for i in range(len(slike)):
        sivaSlika = cv2.cvtColor(slike[i], cv2.COLOR_BGR2GRAY)
        sivaSlikaMedianFilter = median_filter(sivaSlika, 1)
        lap = cv2.Laplacian(sivaSlikaMedianFilter,cv2.CV_64F)
        ostraSlika = cv2.add(sivaSlika,-(k*lap)) 
        bezNeostrina.append(ostraSlika)
    return bezNeostrina

#DODAJ KONTRAST
def dodajKontrast(slika, alfa, beta = 0):
    new_image = cv2.convertScaleAbs(slika, alpha=alfa, beta=beta)
    return new_image
   

#DODAJ OSVJETLJENJE
def dodajOsvjetljenje(slika, beta):
    return dodajKontrast(slika,1,beta)

#UJEDNACAVANJE HISTOGRAMA
def ujednacavanjeHistograma(putanja):
    img_gray_mode = cv2.imread(putanja, cv2.IMREAD_GRAYSCALE)
    equ = cv2.equalizeHist(img_gray_mode)
    return equ

#PODIJELI SLIKE NA TRAIN I TEST
def podijeliNaTrainiTest(slike):
    random.shuffle(slike)
    split = int(0.7 * len(slike))
    slikeTrain = slike[:split]
    slikeTest = slike[split:]
    return slikeTrain, slikeTest 

#PROJEKTNI ZADATAK 2

#KONVERTUJ U GRAYSCALE
def konvertujUGrayscale(slike):
    return cv2.cvtColor(slike, cv2.COLOR_BGR2GRAY)

#UCITAJ SLIKE I RESIZE
def ucitajKropano():
    rootpath = r"E:\ajdin\4. godina\POOS\traffic_sign_recognition_poos-master\Kropano"
    slike = []
    for filename in glob.glob(rootpath + '/*.png'):
        slika = cv2.imread(filename, 0)
        slike.append(cv2.resize(slika,(256,256), interpolation = cv2.INTER_CUBIC))
    return slike


#DESKRIPTOR
def DeskriptorHOG(slike):
    nove = []
    for i in range(len(slike)):
          img = slike[i]
          fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(4, 4), visualize=True, multichannel=False)
          #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
          #ax1.axis('off')
          #ax1.imshow(img, cmap=plt.cm.gray)
          #ax1.set_title('Pocetna slika') 
          #hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
          #ax2.axis('off')
          #ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
          #ax2.set_title('Histogram of Oriented Gradients')
          #plt.show()
          #nove.append(hog_image)
    return nove

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

#DESKRIPTOR
def extract_color_histogram(image, bins=(8, 8, 8)):
    image = ukloniSum(image)
    #image = dodajKontrast(image,20,0)
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,[0, 180, 0, 256, 0, 256])

    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)

	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
    else:
        cv2.normalize(hist, hist)
	# return the flattened histogram as the feature vector)
    return hist.flatten()

def readTrafficSigns2(rootpath = r"E:\ajdin\4. godina\POOS\traffic_sign_recognition_poos-master"):
    images = [] # images
    labels = [] # corresponding labels
    koordinate = []
    # loop over all 42 classes
    for c in range(0,3):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(cv2.imread(prefix + row[0])) # the 1th column is the filename
            x1 = row[3]
            y1 = row[4]
            x2 = row[5]
            y2 = row[6]
            info = x2, y2, x1, y1
            koordinate.append(info)     
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return images, labels, koordinate

#MODEL
def model(sve, treba):
    # grab the list of images that we'll be describing
    print("[INFO] describing images...")
    imagePaths = []


    # initialize the raw pixel intensities matrix, the features matrix,
    # and labels list
    rawImages = []
    features = []
    labels = []
    # loop over the input images
    for i in range(len(sve)):
	    # load the image and extract the class label (assuming that our
	    # path as the format: /path/to/dataset/{class}.{image_num}.jpg
        image = sve[i]
        label = treba[i].split(os.path.sep)[-1].split(".")[0]

	    # extract raw pixel intensity "features", followed by a color
	    # histogram to characterize the color distribution of the pixels
        #pixels = image_to_feature_vector(image)
        hist = extract_color_histogram(image)
	    # update the raw images, features, and labels matricies,
	    # respectively
        #rawImages.append(pixels)
        features.append(hist)
        labels.append(label)


        if i > 0 and i % 10 == 0:
            print("[INFO] processed {}/{}".format(i, len(imagePaths)))

    # show some information on the memory consumed by the raw images
    # matrix and features matrix
    #rawImages = np.array(rawImages)
    features = np.array(features)
    labels = np.array(labels)
    #print("[INFO] pixels matrix: {:.2f}MB".format(
	#    rawImages.nbytes / (1024 * 1000.0)))
    print("[INFO] features matrix: {:.2f}MB".format(
	    features.nbytes / (1024 * 1000.0)))

    # partition the data into training and testing splits, using 75%
    # of the data for training and the remaining 25% for testing
    #(trainRI, testRI, trainRL, testRL) = train_test_split(
	#    rawImages, labels, test_size=0.25, random_state=42)
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	    features, labels, test_size=0.15, random_state=42)
    # train and evaluate a k-NN classifer on the raw pixel intensities
    #print("[INFO] evaluating raw pixel accuracy...")
    #model = KNeighborsClassifier(n_neighbors=20)
    #model.fit(trainRI, trainRL)
    #acc = model.score(testRI, testRL)
    #print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

    # train and evaluate a k-NN classifer on the histogram
    # representations
    print("[INFO] evaluating histogram accuracy...")
    model = KNeighborsClassifier(n_neighbors=20)
    model.fit(trainFeat, trainLabels)
    y_test = model.predict(testFeat)
    acc = model.score(testFeat, testLabels)
    print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))  
    matrica = metrics.confusion_matrix(testLabels, y_test)
    TP = matrica[0, 0]
    TN = matrica[1, 1] + matrica[1, 2] + matrica[2, 1] + matrica[2, 2]
    FP = matrica[0, 1] + matrica[0, 2]
    FN = matrica[1, 0] + matrica[2, 0]
    sensitivity = TP / float(FN + TP)
    print("[INFO] histogram sensitivity: {:.2f}%".format(sensitivity*100))

    specificity = TN / (TN + FP)
    print("[INFO] histogram specificity: {:.2f}%".format(specificity*100))
    
    #eksportovanje modela
    filename = 'TrafficSignRec.sav'
    joblib.dump(model, filename)

    #loadanje modela
    loaded_model = joblib.load(filename)
    result = loaded_model.score(testFeat, testLabels)
    print("[INFO ]histogram accuracy from loaded model: {:.2f}%".format(result * 100))



  
  


slike,labels,koordinate = readTrafficSigns2()
model(slike, labels)

#slike = ucitajKropano()
#DeskriptorHOG(slike)

#spremi(siveslike, r"C:\Users\Admir\Desktop\ProjekatPOOS\SiveSlike\SivaSlika")
#trainImages, trainLabels, koordinate = readTrafficSigns('GTSRB/Training')
#print(len(trainLabels), len(trainImages), len(koordinate))
#kropano = krop(koordinate, trainImages)
#putanja = r"C:\Users\Admir\Desktop\ProjekatPOOS\Kropano\Slika"
#spremi(kropano, putanja)
#putanja2 = r"C:\Users\Admir\Desktop\ProjekatPOOS\BezSuma\SlikaBezSuma"
#bezSuma = ukloniSum(kropano)
#spremi(bezSuma, putanja2)
#putanja3 = r"C:\Users\Admir\Desktop\ProjekatPOOS\BezNeostrina\SlikaBezNeostrina"
#bezNeostrina = ukloniNeostrinu(bezSuma)
#putanja4 = r"C:\Users\Admir\Desktop\ProjekatPOOS\SaKontrastom\SlikaSaKontrastom"
#temp = []
#temp.append(dodajKontrast(kropano[0], 1))
#spremi(temp, putanja4)
#putanja5 = r"C:\Users\Admir\Desktop\ProjekatPOOS\SaOsvjetljenjem\SlikaSaOsvjetljenjem"
#temp2 = []
#temp2.append(dodajOsvjetljenje(kropano[0], 60))
#spremi(temp2, putanja5) 
#putanja6 = r"C:\Users\Admir\Desktop\ProjekatPOOS\SaHistogramom\SlikaSaHistogramom"
#temp3 = []
#temp12 = r"C:\Users\Admir\Desktop\ProjekatPOOS\bmw.jpg"
#temp3.append(ujednacavanjeHistograma(temp12))
#spremi(temp3, putanja6)
#slikeTrain, slikeTest = podijeliNaTrainiTest(trainImages)
#putanja7 = r"C:\Users\Admir\Desktop\ProjekatPOOS\Train\SlikeTrain"
#putanja8 = r"C:\Users\Admir\Desktop\ProjekatPOOS\Test\SlikeTest"
#spremi(slikeTrain, putanja7)
#spremi(slikeTest, putanja8)


#spremi(bezNeostrina, putanja3)


#plt.imshow(kropano[41])
#plt.show()