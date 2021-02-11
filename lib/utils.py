#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 12:01:24 2021

@author: henry
"""
import matplotlib.pyplot as plt
import numpy as np
import imageio

import numpy as np
import matplotlib.pyplot as plt
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color

from skimage.transform import AffineTransform, warp
from skimage.transform import rotate

#segmentar
from skimage.filters import median
from skimage.morphology import disk
from scipy import ndimage
from scipy import ndimage as ndi
from skimage.feature import canny
import pandas as pd
from PIL import Image
import re

import cv2


base='FOTOS_DISCOS_1/'
base='FOTOS_DISCOS_2_MEDIDAS_ANILLOS_Y_PRESENCIAMC/'


def histograma(I, nBins):
    M,N = I.shape
    hist = np.zeros(nBins)
    for i in range(M):
        for j in range(N):
            hist[int(I[i,j])]+= 1

    hist_n = hist/(M*N)
    
    return hist_n

def histogramaAcumulado(I,nBins):
    hist = histograma(I,nBins)
    cdf = np.zeros(nBins)
    
    cdf[0] = hist[0]
    for i in range(1,nBins):
        cdf[i]= hist[i]+cdf[i-1]

    return cdf
def ecualizarHistograma(I):
    L=256
    M,N = I.shape
    cdf = histogramaAcumulado(I,L)
    feq = np.floor(cdf*(L-1))
    I_eq = feq[I].astype(int)
    return I_eq

def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax

def segmentarImagen(imageGray,debug=False):


    I = median(imageGray,disk(5))
    #image_show(I)
    edges = canny(I/255.)
    #fill_coins = ndi.binary_fill_holes(edges)
    if debug:
        image_show(edges)
    
    
    
    # Elemento estructura1 conectividad 8
    struct2 = ndimage.generate_binary_structure(2,2)
    
    iteraciones=6
    
    dil = ndimage.binary_dilation(edges,structure=struct2,iterations=iteraciones).astype(edges.dtype)
    
    #image_show(dil)
    
    # llenar
    
    fill_coins = ndi.binary_fill_holes(dil)
    #image_show(fill_coins)
    
    
    
    
    iteraciones=30
    
    ero = ndimage.binary_erosion(fill_coins,structure=struct2,iterations=iteraciones).astype(fill_coins.dtype)
    dil2 = ndimage.binary_dilation(ero,structure=struct2,iterations=iteraciones-5).astype(edges.dtype)
    
    #image_show(dil2)
    
    
    tronco = np.where(dil2==True)
    
    img_seg = np.zeros(imageGray.shape)   
    img_seg[tronco] = imageGray[tronco]

    return img_seg

def rgbToluminance(img):
        M,N,C = img.shape
        imageGray = np.zeros((M,N))
        imageGray[:,:] = (img[:,:,0]*0.2126 + img[:,:,1]*0.7152 + img[:,:,2]*0.0722).reshape((M,N))

        return imageGray
from scipy.signal import find_peaks



def smoothingProfile(perfil):
    filtered = []
    for i in range(1,len(perfil)-1):
        filtered.append(np.mean(perfil[i-1:i+2]))
    
    return np.array(filtered)
def moving_average(x, w,windows):
    return np.convolve(x, windows, 'valid') / w

from scipy.ndimage import gaussian_filter

def filterSobel(I):
    from scipy import signal
    
    sobelX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobelY = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    
    Ix = signal.convolve2d(I, sobelX, mode ='same', boundary = 'symm')
    Iy = signal.convolve2d(I, sobelY, mode='same',boundary ='symm')
    
    return Ix,Iy


def buscarBorde(copia,angle,end,radio=0,centro=None):
    """
        angulo =  {0,pi/4,pi/2,3pi/4,pi,5pi/4,6pi/4,7pi/4}
        ptosCard= {S, SE , E  , NE  , N, NW  , W   , SW   }
         | 
        ----------->x
         |
         | IMAGEN
         |
         y
         
    """
    i = 0
    M, N = copia.shape
    y_pix =[]
    x_pix = []
    
    background = 0
        
    ctrl = True      
    while ctrl:
        x = centro[1] + i*np.sin(angle)
        y = centro[0] + i*np.cos(angle)
        x = x.astype(int)
        y = y.astype(int)
      

        if i==0 or not (x==x_pix[-1] and y==y_pix[-1]):
            y_pix.append(y)
            x_pix.append(x)

        if end == 'radio':
            r = np.sqrt((x-centro[1])**2 + (y-centro[0])**2)

            if r>radio:
                borde = [y,x]
                ctrl = False
        elif background == copia[y,x]:
                borde = [y,x]
                ctrl = False
        i +=1


    return borde,np.array(y_pix),np.array(x_pix)

def extraerPerfiles(img_seg,centro,debug = False):
    angle = {}
    #oeste
    angle['W'] = -np.pi/2
    #sur
    angle['S'] = 0
    #este
    angle['E'] = np.pi/2
    #norte
    angle['N'] = np.pi
    #noreste
    angle['NE'] = np.pi*3/4
    #noroste
    angle['NW'] = np.pi*5/4
    #suroeste
    angle['SW'] = -np.pi/4
    #sureste
    angle['SE'] = np.pi/4
    
    perfiles = {}
    
    for ptc,angulo in sorted(angle.items()):
        if debug:
            print(f'Punto cardianl {ptc} Angulo {angulo}')
    
        borde,y,x = buscarBorde(img_seg,angulo,end='background',centro=centro)
        perfil = img_seg[y,x]
        perfiles[ptc] = perfil
        #segmento = img_seg[y[::-1],x[0]-100:x[0]+100] 
        if debug: 
            fig, axs = plt.subplots(1)
            fig.suptitle(ptc)
            axs.plot(perfil)
    
    return perfiles



def extraerSubperfiles(perfiles,base,archivo,debug=False):
    """
        extraer los perfiles basados en los radios acumulados medidos manualmente por un experto
    """
    tif = Image.open(base+archivo)
    #image_resolution = 2.54/tif.info['dpi'][0]
    print(tif.info)
    image_resolution = tif.info['dpi'][0]/25.4
    print(tif.info['dpi'])
    print(image_resolution)
    if archivo[0]=='F':
        data = pd.read_csv(base+'fymsa.csv',sep=';')
    else:
        data = pd.read_csv(base+'lumin.csv',sep=';')
    
    
    partido = re.split(r'\.',archivo)
    print(partido)
    dfFoto = data[data['Codigo'] == partido[0]]
    
    radiosReales = {}
    etiquetasTodas = {}
    i=0
    puntoCardinales= ['N','S','E','W']
    
    for i in range(len(puntoCardinales)):
        puntoC = puntoCardinales[i]
        r = dfFoto[f'r{puntoC} mm anual'].values.astype(float)
        etiquetas = dfFoto[f'{puntoC}'].values
        
        r_pix = (r*image_resolution).cumsum().astype(int)
        print(r_pix)
        print(r.cumsum().astype(int))
        perfil = perfiles[puntoC]
    
        if debug:
            plt.figure()
    
        radiosReales[puntoC]=[]
        etiquetasTodas[puntoC] = []
    
        inicio = 0
        for i,fin in enumerate(r_pix):
            if fin<perfil.shape[0]:
                radiosReales[puntoC].append(perfil[inicio:fin])
                etiquetasTodas[puntoC].append(etiquetas[i])
                inicio = fin-1
            
    
    return radiosReales,etiquetasTodas


