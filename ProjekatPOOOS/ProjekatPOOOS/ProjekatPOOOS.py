import matplotlib.pyplot  as plt
import csv
import matplotlib
import random
import cv2
import sys
import numpy as np
from scipy.ndimage.filters import median_filter

#UCITAVA SLIKE SA DATA SETA

def readTrafficSigns(rootpath):
    rootpath = r"C:\Users\Admir\Desktop\ProjekatPOOS"
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
def ukloniSum(slike):
    bezSuma = []
    for i in range(len(slike)):
        bezSuma.append(cv2.fastNlMeansDenoisingColored(slike[i],None,10,10,7,21)) 
    return bezSuma

#UKLANJA NEOSTRINE
def ukloniNeostrinu(slike, k=0.7):
    bezNeostrina = []
    for i in range(len(slike)):
        sivaSlika = cv2.cvtColor(slike[i], cv2.COLOR_BGR2GRAY)
        sivaSlikaMedianFilter = median_filter(sivaSlika, 1)
        lap = cv2.Laplacian(sivaSlikaMedianFilter,cv2.CV_64F)
        ostraSlika = sivaSlika - k*lap
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


trainImages, trainLabels, koordinate = readTrafficSigns('GTSRB/Training')
print(len(trainLabels), len(trainImages), len(koordinate))
kropano = krop(koordinate, trainImages)
putanja = r"C:\Users\Admir\Desktop\ProjekatPOOS\Kropano\Slika"
#spremi(kropano, putanja)
#putanja2 = r"C:\Users\Admir\Desktop\ProjekatPOOS\BezSuma\SlikaBezSuma"
#bezSuma = ukloniSum(kropano)
#spremi(bezSuma, putanja2)
#putanja3 = r"C:\Users\Admir\Desktop\ProjekatPOOS\BezNeostrina\SlikaBezNeostrina"
#bezNeostrina = ukloniNeostrinu(bezSuma)
putanja4 = r"C:\Users\Admir\Desktop\ProjekatPOOS\SaKontrastom\SlikaSaKontrastom"
temp = []
temp.append(dodajKontrast(kropano[0], 1))
spremi(temp, putanja4)
putanja5 = r"C:\Users\Admir\Desktop\ProjekatPOOS\SaOsvjetljenjem\SlikaSaOsvjetljenjem"
temp2 = []
temp2.append(dodajOsvjetljenje(kropano[0], 60))
spremi(temp2, putanja5) 
putanja6 = r"C:\Users\Admir\Desktop\ProjekatPOOS\SaHistogramom\SlikaSaHistogramom"
temp3 = []
temp12 = r"C:\Users\Admir\Desktop\ProjekatPOOS\bmw.jpg"
temp3.append(ujednacavanjeHistograma(temp12))
spremi(temp3, putanja6)
slikeTrain, slikeTest = podijeliNaTrainiTest(trainImages)
putanja7 = r"C:\Users\Admir\Desktop\ProjekatPOOS\Train\SlikeTrain"
putanja8 = r"C:\Users\Admir\Desktop\ProjekatPOOS\Test\SlikeTest"
spremi(slikeTrain, putanja7)
spremi(slikeTest, putanja8)


#spremi(bezNeostrina, putanja3)


plt.imshow(kropano[41])
plt.show()