#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 06:27:04 2020

@author: henry
"""
import utils

import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import imageio
import numpy as np


#segmentar
from skimage.filters import median

import numpy as np
import cv2


from skimage.transform import AffineTransform, warp
from skimage.transform import rotate

ptosCardinales = ['N','NE','E','SE','S','SW','W','NW']
angulos        = [ 0, 45,  90,   135 ,180, -135,-90,-45]
base='base/sinManchas/'


def shift(image, vector):
    transform = AffineTransform(translation=vector)
    shifted = warp(image, transform, mode='wrap', preserve_range=True)
    #image_show(shifted)
    #shifted = shifted.astype(image.dtype)

    return shifted.astype(np.uint8)

def extraerOctantes(image, centro, debug = False):
    global ptosCardinales,angulos
    
    
    Irot = image.copy()
    M,N = Irot.shape
    vector = (-int(M/2)+centro[1],-int(N/2)+centro[0])
    Ishift = shift(Irot, vector)

    octantes = {}
    #rotacion en sentido antihorario
    
    #ptosCardinales = ['NW']
    for i in range(len(ptosCardinales)):
            ancho = 100
            alto = 50
            x,y = int(M/2),int(N/2)
            I = rotate(Ishift, angulos[i], preserve_range=True,order=5).astype(Ishift.dtype)
            

            
            octantes[ptosCardinales[i]] = I[:x-alto,y-ancho:y+ancho] #if borde[0]-alto>0 else I[:x-alto,y-ancho:y+ancho] 
            if debug:
                plt.figure()
                plt.title(ptosCardinales[i])
                plt.imshow(octantes[ptosCardinales[i]],cmap='gray')
                plt.axis('off')


    return octantes

def descriptoresORB(octantes,queryImg_gray,debug=False):

    
    trainImg = octantes.copy()
    trainImg_gray = trainImg 
    #trainImg_gray = Iseg.astype(np.uint8)
    orb = cv2.ORB_create()

    
    #find and compute the descriptors
    kp1,des1 = orb.detectAndCompute(queryImg_gray,None)
    kp2,des2 = orb.detectAndCompute(trainImg_gray,None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    
    if des2 is not None:
        matches = bf.match(des1,des2)
        
        matches = sorted(matches,key = lambda x:x.distance)
    
        valores = []
        
        for i in matches:   
            valores.append(i.distance)
    
        n  = len(matches)

        return n,np.array(valores).mean()
    
    return None,None

def detectarCorteEnOctante(IsegScaled,centro,debug=False):   
    global ptosCardinales,base
    resultado = []
    # 1.0 extraer octantes
    octantes = extraerOctantes(IsegScaled,centro, debug)
            
    queryImg_gray = imageio.imread(base+'F4B_cortada3.tif')
    noRaja_gray = imageio.imread(base + 'sinRaja.png')
    #queryImg_gray = imageio.imread(base+'raja.png')
    canT,medias_distancias = [],[]

    for key in octantes:
        #oct_eq = utils.ecualizarHistograma(octantes[key].copy())
        n_r, media_r = descriptoresORB(octantes[key],queryImg_gray)
        if n_r:
            canT.append(n_r)
            medias_distancias.append(media_r)
        n_nr, media_nr = descriptoresORB(octantes[key],noRaja_gray)
        if n_nr:
            #canT.append(n_r)
            #medias_distancias.append(media_r)
            if debug:
                print(f' {key}  {n_r} {1.2*n_nr} {n_nr}')
            if n_r>1.2*n_nr:
                resultado.append(key)
            

    return resultado

#%%

if __name__=="__main__":    
    base='../base/sinManchas/'
    filename = ['F10A.tif','F10B.tif','F2Ab.tif','F2B.tif','F4A.tif','F4A_rot.tif']
    centros =  [ [888,779],[850,968],[1208,1300],[1039,1068],[1033,878],[1019,826] ]

    indice = 1 #TP

    img_orig = imageio.imread(base+filename[indice])
    
    centro = centros[indice]
    
    # #1.0 rgb2gray
    image = color.rgb2gray(img_orig)*255 
    
    image = image.astype(np.uint8)
    
    img_seg = utils.segmentarImagen(image)
    
    debug = True
    if debug:
        utils.image_show(im)
    print(detectarCorteEnOctante( img_seg, centro, debug = debug))
        














