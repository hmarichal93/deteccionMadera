#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 17:48:47 2021

@author: henry
"""
import numpy as np

def media(X,color=None):
    media = []
    for i in range(len(X)):
        if color:
            r = X[i][0].mean()
            g = X[i][1].mean()
            b = X[i][2].mean()
            media.append([r,g,b])
            med = np.array(media).reshape((-1,3))
        else:
            media.append(X[i].mean())
            med = np.array(media).reshape((-1,1))
    
    return med

def desviacion(X,color=None):
    desviacion = []
    for i in range(len(X)):
        if color:
            r = X[i][0].std()
            g = X[i][1].std()
            b = X[i][2].std()
            desviacion.append([r,g,b])
            std = np.array(desviacion).reshape((-1,3))
              
        else:
            desviacion.append(X[i].std())
            std = np.array(desviacion).reshape((-1,1))
    
    return std

def fit_pol(X,color=None):
    pol = []
    removed = []
    for i in range(len(X)):
        try: 
            if color:
                r = np.polyfit(np.linspace(0, len(X[i][0])-1,len(X[i][0])),X[i][0],1)
                g = np.polyfit(np.linspace(0, len(X[i][0])-1,len(X[i][0])),X[i][1],1)
                b = np.polyfit(np.linspace(0, len(X[i][0])-1,len(X[i][0])),X[i][2],1)
                pol.append([r,g,b])        
            else:
                ordenadas = np.linspace(0, len(X[i])-1,len(X[i]))
                if ordenadas.shape[0]>0:
                    pol.append(np.polyfit(ordenadas,X[i],1))
                    
                else:
                    print(f"{ ordenadas} {i}")
        except:
            
            print("excepcion elemento {i}")
            removed.append(i)
            pol.append([1,1])
    return np.array(pol), removed 

def proporcion(X,color=None):
    prop = []
    for i in range(len(X)):
        media = X[i].mean()
        arriba = len(np.where(X[i]>media*1.1)[0])
        
        abajo = len(np.where(X[i]<media*1.1)[0])
        #print(f'{abajo} {arriba}')
        
        if abajo>0:
            prop.append(arriba/abajo)
        else:
            prop.append(0)
    
    
    
    return np.array(prop).reshape((-1,1))