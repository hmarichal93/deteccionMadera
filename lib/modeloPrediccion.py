#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 11:49:33 2021

@author: henry
"""
#bibliotecas propias
import utils 
import features

#bibliotecas externas
import imageio 
import numpy as np
import pandas as pd
import re 
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import pickle

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

        global ptosCardinales  
        global segmento
        
        arreglo = np.array(coords)
        radios = np.sort(arreglo[:,0]).astype(int)
        inicio = radios[0]
        j = 0
        etiquetado = input('Ingrese etiquetas: ')
        
        etiquetado = etiquetado.split(',')
        etiquetado = np.array(etiquetado).astype(int)
        print(etiquetado)
        print(radios)
        #subPerfil = {}
        #subPerfil[str(idx)] = []
        for i,fin in enumerate(radios[1:]):
            print(i)
            subPerfil.append([perfiles[inicio:fin],etiquetado[j]])
            inicio = fin 
            j +=1

        

        




def loop(perfil):
    global perfiles
    global ptosCardinales

    global segmento
    
    fig = plt.figure()
    if segmento:
        perfiles = perfil
    ax = fig.add_subplot(111)
   
    ax.plot(perfil)
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    fig.canvas.mpl_connect('close_event', handle_close)
    plt.show()
    return fig

def handle_close_crear_base(evt):
        global coords
        global subPerfil
        global perfiles
        global idx
        global ptosCardinales  
        global segmento
        global databaseName
        print("CreandoBase")

        f = open(databaseName, "a")
        #f.write("Now the file has more content!")
        
        
        for radio,label in subPerfil:
            if radio.shape[0]>0:
                 string =";".join(radio.astype(str))
                 string = string+f"|{label}\n"
                 f.write(string)
        f.close()
    
def seleccionarSegmento(img_seg):
    global fig,cid
    
    fig = plt.figure()
    plt.imshow(img_seg,cmap='gray')
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick_endpoints_segmento)
    fig.canvas.mpl_connect('close_event', handle_close_crear_base)
    plt.show()
    
def onclick_endpoints_segmento(event):
    global ix, iy
    global coordsSeg,cid,fig,img_seg, coordenadas,coords
    ix, iy = event.xdata, event.ydata
    print('x = %d, y = %d'%(ix, iy))

    coordsSeg.append((ix, iy))
    
    
    if len(coordsSeg) == 2:
        calcularSegmento()
        #fig.canvas.mpl_disconnect(cid)
        loop(img_seg[np.array(coordenadas)[:,0],np.array(coordenadas)[:,1]])
        coordsSeg = []
        coordenadas = []
        coords = []

    return coordsSeg    


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
            if 1> np.abs((m*x_i + n - y_i)): 
                    coordenadas.append([y_i,x_i])
                                
    
    return coordenadas

#%%
debug = True
#base='FOTOS_DISCOS_2_MEDIDAS_ANILLOS_Y_PRESENCIAMC/'
#filename = ['L4Ab.tif','F2Ab.tif','F10A.tif','L7A.tif','L4A.tif','F10B.tif','F4B.tif','F2A.tif','F2B.tif']
#centros =[[1189,1255],[1208,1300],[888,779],[1062,1123],[1061,1061],[850,968],[1038,1068],[902,880],[1030,1067],[1004,990]] 
databaseName = "baseDeDatosAMano.txt"
databaseBig = "../modelos/datosCrudosReducido.txt"
base='../base/sinManchas/'
filename = ['F10A.tif','F10B.tif','F2Ab.tif','F2B.tif','F4A.tif','F4A_rot.tif']
centros =  [ [888,779],[850,968],[1208,1300],[1039,1068],[1033,878],[1019,826] ]
#indice=1 no sirve para base
#indice = 6 no tiene dpi sino resolucion
indice = 0
centro = centros[indice]

#%%
## Cargar imagen tronco
img = imageio.imread(base+filename[indice])

#%%
## Extraer los 4 perfiles
imageGray = utils.rgbToluminance(img)
img_seg = utils.segmentarImagen(imageGray)
debug = False
if debug:
    utils.image_show(img_seg)


#%%
## Extraer los subPerfiles (perfil por radio)
perfiles = utils.extraerPerfiles(img_seg, centro,debug=True)

#%%
debug = True
segmento = True
automatico = True

if automatico:
    archivo = filename[indice]
    subRadios,etiquetas = utils.extraerSubperfiles(perfiles,base,archivo)
    
    if debug:
        for pto in ['N','E','S','W']:
            plt.figure()
            plt.title(pto)
            inicio = 0
            for idx,radio in enumerate(subRadios[pto]):
                fin = radio.shape[0]
                x = np.linspace(inicio,inicio+fin-1,fin)
                plt.plot(x,radio)
                inicio += fin-1
else:
    #Sirve para la implementación de testing
    subPerfil = []
    coordsSeg = []
    coordenadas = []
    coords = []
    seleccionarSegmento(img_seg)
            
#%%
## Guardaren archivo csv
creando = True

if creando:
    f = open("datosCrudosReducido.txt", "a")
    #f.write("Now the file has more content!")
    radio_minimo = 2
    radio_maximo = 13
    for pto in ['N','E','S','W']:
        cantidad = len(subRadios[pto][radio_minimo:radio_maximo])
        for idx, radio in enumerate(subRadios[pto]):
            if radio.shape[0]>0:
                  string =";".join(radio.astype(str))
                  string = string+f"|{etiquetas[pto][idx]}\n"
                  f.write(string)
    f.close()
    
#%% convertir datos crudos a features

datos = []
labels = []
f =open(databaseBig, "r")

lines = f.readlines()

for line in lines:
    string = re.split(r'\|',line)
    
    radio = np.array(re.split('\;',string[0])).astype(float)
    if radio.shape[0]>15:
        #print(radio)
        datos.append(radio)
        etiqueta = int(string[1])
        labels.append(etiqueta)
    #print(etiqueta)
f.close()

color = False
X_features = features.proporcion(datos,color)
X_features = np.hstack((X_features,features.media(datos,color)))
X_features = np.hstack((X_features,features.desviacion(datos,color)))
cantidad = 2
feat,exp = features.fit_pol(datos,color)
if len(exp)==0:
    print("bien")
    X_features = np.hstack((X_features,feat))

#%% train model

clf = RandomForestClassifier(random_state=0)


    
scores = cross_val_score(clf, X_features, np.array(labels), cv=3,scoring='f1')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() ))
model = clf.fit(X_features,np.array(labels))

#%% save model
# save the model to disk
filename = 'finalized_model_bigDatabase.sav'
pickle.dump(model, open(filename, 'wb'))

