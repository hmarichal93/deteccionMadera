#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 16:26:11 2020

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

import pickle


import lib.utils as utils
import lib.features as features
import lib.rajaduras as rajaduras

#%%


def onclick(event):
    global ix, iy
    global coords
    ix, iy = event.xdata, event.ydata
    print('x = %d, y = %d'%(ix, iy))

    coords.append((ix, iy))

    return coords


def handle_close(evt):
        global coords
        global subPerfil
        global perfiles
        global idx
        global ptosCardinales  
        global segmento
        
        arreglo = np.array(coords)
        radios = np.sort(arreglo[:,0]).astype(int)
        inicio = radios[0]
        if segmento:
            subPerfil = {}
            subPerfil['1'] = []
            for i,fin in enumerate(radios[1:]):
                subPerfil['1'].append(perfiles[inicio:fin])
                inicio = fin 
            
            predecir()
        else:
            subPerfil[ptosCardinales[idx]] = []
            for i,fin in enumerate(radios[1:]):
                subPerfil[ptosCardinales[idx]].append(perfiles[ptosCardinales[idx]][inicio:fin])
                inicio = fin
            idx +=1
            if idx<len(ptosCardinales):
                loop(perfiles[ptosCardinales[idx]],ptosCardinales[idx])
            else:
                predecir()



def loop(perfil,title = None):
    global perfiles
    global ptosCardinales
    global idx
    global segmento
    global semiAutomatic
    global subPerfil
    

    
    if not semiAutomatic:
        fig = plt.figure()
        if segmento:
            perfiles = perfil
        ax = fig.add_subplot(111)
        plt.title(title)
        ax.plot(perfil)
        ax.axis('off')
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
        fig.canvas.mpl_connect('close_event', handle_close)
    else:
        debug = True
        subPerfil = {}
        key='Segmento'
        #radios = {}
    
        #perfil = perfiles[key]    
        #filtrar señal para eliminar ruido              
        w = 10
        windows = np.hanning(w)
        profileFiltered2 =  utils.moving_average(perfil,w,windows)
        
        #determinar minimos
        xx =-1*profileFiltered2
        peaks, _ = utils.find_peaks(xx,prominence=1,distance=40)
        
        y = perfil[int(w/2)-1:-int(w/2)]
        x = np.linspace(int(w/2),len(y)-1+int(w/2),len(y))
        radios = x[np.array(peaks)].copy().astype(int)
        print(radios)
        if debug:
            plt.figure()
            plt.title(f'Perfil {key}')
            plt.plot(perfil)
            yy = perfil[np.array(peaks)+int(w/2)]
            plt.plot(radios,yy, "x")
        
    
        inicio = radios[0]
        subPerfil[key] = []
        
        for i,fin in enumerate(radios[1:]):
            subPerfil[key].append(perfil[inicio:fin])
            inicio = fin
                

        predecir()
    return 0


def predecir():
    global subPerfil,modelName
    datos = []
    # load the model from disk
    loaded_model = pickle.load(open(modelName, 'rb'))
    
    for key in subPerfil.keys():
        datos = []
        for radio in subPerfil[key]:
            datos.append(radio)
    
    
        color = False
        X_features = []
        X_features = features.proporcion(datos,color)
        X_features = np.hstack((X_features,features.media(datos,color)))
        X_features = np.hstack((X_features,features.desviacion(datos,color)))
    
        feat,exp = features.fit_pol(datos,color)
        
        if len(exp)==0:
            
            X_features = np.hstack((X_features,feat))
    
        
            predicted = loaded_model.predict(X_features)
            
            print(f'Radio {key} prediccion {predicted}')
        else:
            
            print('ocurrio un error')
    
def seleccionarSegmento(img_seg):
    global fig,cid
    
    fig = plt.figure()
    plt.imshow(img_seg,cmap='gray')
    cid = fig.canvas.mpl_connect('button_press_event', onclick2)

def onclick2(event):
    global ix, iy
    global coordsSeg,cid,fig,img_seg
    ix, iy = event.xdata, event.ydata
    print('x = %d, y = %d'%(ix, iy))

    coordsSeg.append((ix, iy))
    
    
    if len(coordsSeg) == 2:
        coordenadas= calcularSegmento()
        fig.canvas.mpl_disconnect(cid)
        loop(img_seg[np.array(coordenadas)[:,0],np.array(coordenadas)[:,1]][::-1],'Segmento')
        
        

    return coordsSeg    

#%%
def calcularSegmento():
    print("dentro de calcular segmento")
    
    global coordsSeg,img_seg,coordenadas
    x1 = int(coordsSeg[0][0])
    y1 = int(coordsSeg[0][1])
    
    x2 = int(coordsSeg[1][0])
    y2 = int(coordsSeg[1][1])

    #algoritmo2

    m = (y1-y2)/(x1-x2)
    n = y1-m*x1

    M,N = img_seg.shape
    x_max = np.maximum(x1, x2)
    x_min = np.minimum(x1, x2)
    y_max = np.maximum(y1, y2)
    y_min = np.minimum(y1, y2)    
    for x_i in range(x_min,x_max+1):
        for y_i in range(y_min,y_max+1): 
            if 1>= np.abs((m*x_i + n - y_i)): 
                    coordenadas.append([y_i,x_i])
                                
    print(len(coordenadas))
    return coordenadas
        


#%% 
if __name__=="__main__":  
    global coordsSeg
    ptosCardinales = ['N','NE','E','SE','S','SW','W','NW']
    debug = False
    segmento = False
    semiAutomatic = True

    modelName = 'modelos/finalized_model_smallBase.sav'
    #modelName = 'finalized_model_bigDatabase.sav'

    base='base/sinManchas/'
    filename = ['F10B.tif','F4A_rot.tif']
    centros = [[850,968],[1019,826]]
    indice = 1
    centro = centros[indice]
    img = imageio.imread(base+filename[indice])
    
    imageGray = utils.rgbToluminance(img)
    img_seg = utils.segmentarImagen(imageGray,debug)
    #utils.image_show(img_seg)
    coordsSeg = []
    coordenadas = []
    coords = []
    if segmento:
        seleccionarSegmento(img_seg)
    else:

        if debug:
            utils.image_show(img_seg)
        
        #% hay rajadura en algunos de los octantes
        debug = False
        rajas = rajaduras.detectarCorteEnOctante( img_seg, centro, debug = debug)
        if rajas:
            print(f'En los  siguientes ptos se tienen rajaduras {rajas}')

            ptosCardinales.pop(ptosCardinales.index(rajas[0]))
        perfiles = utils.extraerPerfiles(img_seg, centro)

            
        if not semiAutomatic:       
            print("Semi Automatic")
            
            subPerfil = {}
            idx = 0        
            loop(perfiles[ptosCardinales[idx]],ptosCardinales[idx])
        else:

            utils.image_show(img_seg)
            debug = True
            subPerfil = {}
            #radios = {}
            for key in ptosCardinales:
                perfil = perfiles[key]    
                #filtrar señal para eliminar ruido              
                w = 10
                windows = np.hanning(w)
                profileFiltered2 =  utils.moving_average(perfil,w,windows)
                
                #determinar minimos
                xx =-1*profileFiltered2
                peaks, _ = utils.find_peaks(xx,prominence=1,distance=30)
                
                y = perfil[int(w/2)-1:-int(w/2)]
                x = np.linspace(int(w/2),len(y)-1+int(w/2),len(y))
                radios = x[np.array(peaks)].copy().astype(int)
                
                if debug:
                    plt.figure()
                    plt.title(f'Perfil {key}')
                    plt.plot(perfil)
                    yy = perfil[np.array(peaks)+int(w/2)]
                    plt.plot(radios,yy, "x")
                
            
                inicio = radios[0]
                subPerfil[key] = []
                
                for i,fin in enumerate(radios[1:]):
                    subPerfil[key].append(perfil[inicio:fin])
                    inicio = fin
                    

            predecir()

            
                
