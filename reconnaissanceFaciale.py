#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Created on Wed Apr 14 14:13:42 2021
#@author: cytech



## Auteur : Louve LE BRONEC, Jenna STORCHI
## 2020-2021 - ING1 GMI
## Reconnaissance faciale



import glob
import numpy as np
import numpy
from matplotlib.pyplot import imread
from matplotlib.pyplot import imsave
import matplotlib.pyplot
from PIL import Image
#import scipy



# Acquisition des images
# pngs : tableau d'images
#pngs = glob.glob('BDD1/FG1/*.pgm')
pngs = glob.glob('NewTests/gallery/*.png')



# Lecture de chaque image
#Lecture de la première ligne
image = imread(pngs[0], True)
heigh = len(image[:,0])
length = len(image[0])



# 4 - Phase d'apprentissage



 #Initialisation de n à 100
n=23



# Question 1-2 : Transformation de chaque matrice en une ligne de taille 4096
# et regroupement dans une matrice de taille 100 x 4096
X = np.array([imread(i,True).flatten()for i in pngs])[range(n),:]



# Question 3 : Centrage et réduction de la matrice
#Matrice des moyennes
tabMoyenne = np.mean(X,0)



#Matrice des écarts types
tabEcartType = np.nanstd(X,0)



#Il faut centrer-réduire X
Y = (X - tabMoyenne)/ tabEcartType



#Matrice 2D réunissant la table des moyennes et la table des écarts types
M = []
M.append(tabMoyenne)
M.append(tabEcartType)



#C = np.dot(np.transpose(Y), Y)



# Question 4
#Décomposition en valeur singulière
U, sigma, V = np.linalg.svd(Y, full_matrices=True)



# On transforme sigma en matrice diagonale
sigma = np.diag(sigma)



#Initialisation des matrices P et D
P=V
D=np.dot(np.transpose(sigma), sigma)



# Question 5 : On cherche la valeur de k



#Initialisation des variables
somme = 0
sommeK = 0
k = 0



#Calcul de la somme de tous les éléments propres
for i in range (0, len(sigma)):
    somme = somme + D[i][i]



#Initialisation d'epsilon
eps = sommeK/somme



#Tant que l'inertie est inférieure à 97%
while eps < 0.97:
    #On somme les valeurs propres
    sommeK = sommeK + D[k][k]

    #On calcule l'inertie actuelle
    eps = sommeK/somme

    #Incrémentation de k
    k = k + 1



# Question 6 : Création de la matrice Pk à partir de P
#On ne conserve que les k premières colonnes
Pk = P[:,range(k)]



#Composantes de chaque image dans la base partielle Pk ;
Z = np.dot(Y, Pk)



# 5 - Base de données à dimension réduite
#Transfomation de la matrice de pixels en une ligne de xm
xm=imread(pngs[1],True).flatten()



#Centrage et réduction de xm
ym=(xm-tabMoyenne)/tabEcartType



#Initialisation de zm
zm = np.dot(ym, Pk)



# 6 - Reconnaissance faciale



# Acquisition des images
# pngs : tableau d'images
pngs2 = glob.glob('BDD2/FG2/*.pgm')



# Lecture de chaque image
#Lecture de la première ligne
image2 = imread(pngs2[0], True)
heigh2 = len(image2[:,0])
length2 = len(image2[0])



#Question 1 : Transformation de la matrice 64 x 64 en une ligne 4096
#xp = np.array([imread(i,True).flatten()for i in pngs2])
xp = np.array(imread("NewTests/a_tester/zach1.png", True).flatten()) # recup image a traiter



#AFFICHAGE DANS LES PLOTS#
#Ouverture d'un bloc figure
tofind = matplotlib.pyplot.figure(1)
#Redimension de l'image à traiter
matplotlib.pyplot.imshow(xp.reshape(heigh,length))
#Titre (parce qu'apparemment ça fait plus propre)
matplotlib.pyplot.title("Individu A Identifier")
#Nuance de gris
matplotlib.pyplot.gray()
########################



#Question 2 : Centrage et réduction de xp
#Matrice des moyennes
tabMoyenneXP = np.mean(xp,0)



#Matrice des écarts types
tabEcartTypeXP = np.nanstd(xp,0)



#Il faut centrer-réduire X
yp = (xp - tabMoyenne)/ tabEcartType



#Question 3 : Coordonnées de cette image dans la base des composantes principales retenues
zp = np.dot(yp, Pk)



#Question 4 : Calcul des distances entre la nouvelle image et celles de la BDD
# di = ||zp - Zi||_2
distance = np.sqrt(((zp-Z)**2).sum(axis=1) )



#Question 5 : Identification du visage par la recherche de la distance minimale.



#Calcul de l'indice de l'image
indiceImg = np.argmin(distance)
#image identifiee dans gallery
print ("Image n° : ",str(indiceImg))



#Distance minimale
mindist= distance[indiceImg]
print ("Distance min : ", str(mindist))



#Ouverture d'un deuxième bloc figure
found=matplotlib.pyplot.figure(2)
#Redimension de l'image du candidat potentiel
matplotlib.pyplot.imshow(X[indiceImg].reshape(heigh,length))
#Titre (parce qu'apparemment ça fait plus propre)
matplotlib.pyplot.title("Candidat potentiel")
#Nuance de gris
matplotlib.pyplot.gray()
