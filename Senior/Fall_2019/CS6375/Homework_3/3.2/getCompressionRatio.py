from __future__ import division
import os
import numpy as np

def mean(arr):
    return sum(arr) / len(arr)
def variance(arr):
    return sum((np.array(arr) - mean(arr))**2) / (len(arr) - 1)

def getKRatio(image):
    n=100
    string=''
    K=[2,5,10,15,20]
    kratioavg=[]
    kratiovar=[]
    for k in K:
        k = str(k)
        print 'Computing k=' + k
        kratio=[]
        for i in range(0,n):
            os.system('java KMeans ' + image + ' ' + k + ' newimage.jpg')
            os.system('zip -qq size.zip newimage.jpg')
            size = int(os.popen('stat --printf="%s" newimage.jpg').read())
            compression = int(os.popen('stat --printf="%s" size.zip').read())
            kratio += [size / compression]
        kratioavg += [mean(kratio)]
        kratiovar += [variance(kratio)]
    return (K, kratioavg, kratiovar)

def printImgCmpRatio(image):
    K, kratioavg, kratiovar = getKRatio(image)
    print 'IMAGE COMPRESSION INFO FOR ' + image
    for i, k in enumerate(K):
        k = str(k)
        avg = str(kratioavg[i])
        var = str(kratiovar[i])
        print 'k=' + k + ' has average compression ' + avg + ' and variance ' + var

printImgCmpRatio('Koala.jpg')
printImgCmpRatio('Penguins.jpg')
