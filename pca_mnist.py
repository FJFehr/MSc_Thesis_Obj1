#
# Fabio Fehr
# 10 June 2020

import open3d as o3d
import numpy as np
import os

from keras.datasets import mnist
from keras.layers import Input, Dense
from keras import regularizers, models, optimizers
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from keras.datasets import mnist

######################################################################################################################
# PCA function #######################################################################################################
######################################################################################################################

# Analytical PCA
def AnalyticalPCA(y, dimension):
    pca = PCA(n_components=dimension)
    return pca.fit(y)
    # loadings = pca.components_
    # return loadings

#  components_ : array, shape (n_components, n_features)
#   returns the  Principal axes in feature space, representing the directions of maximum variance in the data
# explained_variance_ : array, shape (n_components,)
#   The amount of variance explained by each of the selected components.
# singular_values_ : array, shape (n_components,)
#   The singular values corresponding to each of the selected components.
# mean_ : array, shape (n_features,)

######################################################################################################################
# Little MNIST example ###############################################################################################
######################################################################################################################

# load the data
(y, _), (_, _) = mnist.load_data()

# 60000 examples of images in y, each 28 by 28
shape_y = y.shape # (60000, 28, 28)

# reshape y to be a 2D matrix of the dataset (scale?)
# 60000 examples of images in y,784
y = np.reshape(y,[shape_y[0],shape_y[1]*shape_y[2]]).astype('float32')/255 # (60000, 784)

# Now we have squeezed those 60000 down to 16 images of principal components
p_analytical = AnalyticalPCA(y,16).components_ # has dimensions (16, 784)

# reshape before plotting
p_analytical = np.reshape(p_analytical,[16,shape_y[1],shape_y[2]]) # has dimensions (16, 28, 28)

# plot the principal components
def PlotResults(p,dimension,name):
    sqrt_dimension = int(np.ceil(np.sqrt(dimension)))
    plt.figure()
    for i in range(p.shape[0]):
        plt.subplot(sqrt_dimension, sqrt_dimension, i + 1)
        plt.imshow(p[i, :, :],cmap='gray')
        plt.axis('off')
    plt.show()

#PlotResults(p_analytical,16,'AnalyticalPCA')

# pcaMean = AnalyticalPCA(y,16).mean_
# pcaMean = np.reshape(pcaMean,[1,shape_y[1],shape_y[2]]) # has dimensions (1, 28, 28)
#
# # This plot the mean shape
# plt.figure()
# plt.imshow(pcaMean[0, :, :], cmap='gray')
# plt.axis('off')
# plt.show()

# plot the first image (Its a 5)
firstImage =  np.reshape(y[0, :],[1,shape_y[1],shape_y[2]])
plt.figure()
plt.imshow(firstImage[0, :, :], cmap='gray')
plt.axis('off')
plt.show()

# plot the pca reconstruction for the first image

# Check out https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com

# n = 60000, p = 784, k =16
# X(n,p), V(p,k)
# reconstruction = PC scores x Eigen vectors transposed + Mean
# reconstruction = X V V^T + Mean
# dimension X and Mean : 1x784 , V dimension (16, 784)
pca = AnalyticalPCA(y, 16)
mean = pca.mean_  # (784,) meaning a vector
eigenvals = pca.singular_values_

X = y # meaning (60000, 784)
V = pca.components_.T # (784, 16)
b = np.dot(X, V)# PC Scores or shape parameters 60000 x 16
#VVt = np.dot(V, V.T) # (784, 784)

hat_x = np.dot(b, V.T) + mean

reconstructionImage = np.reshape(hat_x,[60000,shape_y[1],shape_y[2]])
plt.figure()
plt.imshow(reconstructionImage[0, :, :], cmap='gray')
plt.axis('off')
plt.show()

# now lets plot the shape parameters against one another.
plt.scatter(b[:,0], b[:,1], alpha=0.5)
plt.title('Scattergram of shape parameters')
plt.xlabel('b1')
plt.ylabel('b2')
#unhelpful in this context Very much between the bounds
#plt.xlim(-3*np.sqrt(eigenvals[0]),3*np.sqrt(eigenvals[0]))
#plt.ylim(-3*np.sqrt(eigenvals[1]),3*np.sqrt(eigenvals[1]))
plt.show()

#plot scattergrams of shape parameters.
def PlotResultsB(b,dimension):
    sqrt_dimension = int(np.ceil(np.sqrt(dimension)))
    plt.figure()
    for i in range(b.shape[1]):
        plt.subplot(sqrt_dimension, sqrt_dimension, i + 1)
        plt.scatter(b[:,0], b[:,i], alpha=0.5)
    plt.show()

PlotResultsB(b,16)

# How many nodes
plt.plot(np.cumsum(pca.explained_variance_)/ sum(pca.explained_variance_))
plt.ylabel('explained_variance')
plt.show()

# extreme values in modes of variation

X = y[0,:] # meaning (60000, 784)
V = pca.components_.T # (784, 16)
extremePosb = 3*np.sqrt(eigenvals[0]) * np.ones((1,16))
extremeNegb = -3*np.sqrt(eigenvals[0]) * np.ones((1,16))


hat_x_pos = np.dot(extremePosb, V.T) + mean
hat_x_neg = np.dot(extremeNegb, V.T) + mean

reconstruction_hat_x_pos = np.reshape(hat_x_pos,[1,shape_y[1],shape_y[2]])
plt.figure()
plt.imshow(reconstruction_hat_x_pos[0, :, :], cmap='gray')
plt.axis('off')
plt.show()

reconstruction_hat_x_neg = np.reshape(hat_x_neg,[1,shape_y[1],shape_y[2]])
plt.figure()
plt.imshow(reconstruction_hat_x_neg[0, :, :], cmap='gray')
plt.axis('off')
plt.show()
