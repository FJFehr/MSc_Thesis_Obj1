# This script works through PCA for the MNIST dataset
# Fabio Fehr
# 10 June 2020

import numpy as np
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
# MNIST example ######################################################################################################
######################################################################################################################

dimension = 16
# load the data
(y, _), (_, _) = mnist.load_data()

# 60000 examples of images in y, each 28 by 28
shape_y = y.shape # (60000, 28, 28)

# reshape y to be a 2D matrix of the dataset (scale?)
# 60000 examples of images in y,784
y = np.reshape(y,[shape_y[0],shape_y[1]*shape_y[2]]).astype('float32')/255 # (60000, 784)

# Now we have squeezed those 60000 down to 16 images of principal components
p_analytical = AnalyticalPCA(y,dimension).components_ # has dimensions (16, 784)

# reshape before plotting
p_analytical = np.reshape(p_analytical,[dimension,shape_y[1],shape_y[2]]) # has dimensions (16, 28, 28)

# plot the principal components
def PlotResults(p,dimension):
    sqrt_dimension = int(np.ceil(np.sqrt(dimension)))
    plt.figure()
    for i in range(p.shape[0]):
        plt.subplot(sqrt_dimension, sqrt_dimension, i + 1)
        plt.imshow(p[i, :, :],cmap='gray')
        plt.title(str(i + 1))
        plt.axis('off')
    plt.savefig('pictures/PCA_components_dim' + str(dimension) + '.png')
    plt.show()

PlotResults(p_analytical,dimension)

# Plot the mean digit

pcaMean = AnalyticalPCA(y,dimension).mean_
pcaMean = np.reshape(pcaMean,[1,shape_y[1],shape_y[2]]) # has dimensions (1, 28, 28)

plt.figure()
plt.imshow(pcaMean[0, :, :], cmap='gray')
plt.axis('off')
plt.show()

# plot the first image (Its a 5)

firstImage =  np.reshape(y[0, :],[1,shape_y[1],shape_y[2]])
plt.figure()
plt.title("Reconstruction")
plt.imshow(firstImage[0, :, :], cmap='gray')
plt.axis('off')
plt.savefig('pictures/PCA_dim' + str(dimension) + '.png')
plt.show()

# plot the pca reconstruction for the first image

# Check out https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com

# n = 60000, p = 784, k =16
# X(n,p), V(p,k)
# reconstruction = PC scores x Eigen vectors transposed + Mean
# reconstruction = X V V^T + Mean
# dimension X and Mean : 1x784 , V dimension (16, 784)
pca = AnalyticalPCA(y, dimension)
mean = pca.mean_  # (784,) meaning a vector
eigenvals = pca.singular_values_

X = y                     # meaning (60000, 784)
V = pca.components_.T     # (784, 16)
b = np.dot(X, V)          # PC Scores or shape parameters 60000 x 16

hat_x = np.dot(b, V.T) + mean

reconstructionImage = np.reshape(hat_x, [60000, shape_y[1], shape_y[2]])
plt.figure()
plt.title("Reconstruction")
plt.imshow(reconstructionImage[0, :, :], cmap='gray')
plt.axis('off')
plt.savefig('pictures/PCA_dim' + str(dimension) + '.png')
plt.show()

######################################################################################################################
# Scattergram ########################################################################################################
######################################################################################################################

# This is important to see if there are any non-linear dependencies in the data.

# now lets plot the shape parameters against one another.
plt.scatter(b[:,0], b[:,1], alpha=0.5, s=0.5)
plt.title('Scattergram of shape parameters')
plt.xlabel('b1')
plt.ylabel('b2')
#unhelpful in this context Very much between the bounds
#plt.xlim(-3*np.sqrt(eigenvals[0]),3*np.sqrt(eigenvals[0]))
#plt.ylim(-3*np.sqrt(eigenvals[1]),3*np.sqrt(eigenvals[1]))
plt.show()

#plot scattergrams of shape parameters.
def PlotResultsB(b,num_of_modes = dimension):
    plt.figure()
    fig_count = 1
    for i in range(num_of_modes):
        for j in range(num_of_modes):
            plt.subplot(num_of_modes, num_of_modes, fig_count)
            plt.scatter(b[:,i], b[:,j], alpha=0.2, s=0.5)
            fig_count += 1
    plt.show()

PlotResultsB(b,4)

######################################################################################################################
# How many nodes required?  ##########################################################################################
######################################################################################################################

#TODO: Calculate PCA for as many nodes as possible then use var explained to determine the dimension

plt.plot(np.cumsum(pca.explained_variance_)/ sum(pca.explained_variance_))
plt.ylabel('explained_variance')
plt.show()

######################################################################################################################
# Modes of variation  ##########################################################################################
######################################################################################################################

# Here we will plot the modes of variation at their extremes -3sqrt(lamba) and 3sqrt(lambda)

# X = y[0,:] # meaning (60000, 784)
# V = pca.components_.T # (784, 16)
# extremePosb = 3*np.sqrt(eigenvals[0]) * np.ones((1,dimension))
# extremeNegb = -3*np.sqrt(eigenvals[0]) * np.ones((1,dimension))
#
# hat_x_pos = np.dot(extremePosb, V.T) + mean
# hat_x_neg = np.dot(extremeNegb, V.T) + mean
#
# # Positive extreme plot
# reconstruction_hat_x_pos = np.reshape(hat_x_pos,[1,shape_y[1],shape_y[2]])
# plt.figure()
# plt.imshow(reconstruction_hat_x_pos[0, :, :], cmap='gray')
# plt.axis('off')
# plt.show()
#
# # Negative extreme plot
# reconstruction_hat_x_neg = np.reshape(hat_x_neg,[1,shape_y[1],shape_y[2]])
# plt.figure()
# plt.imshow(reconstruction_hat_x_neg[0, :, :], cmap='gray')
# plt.axis('off')
# plt.show()

def PlotModesVaration(mean=mean,eigenvals=eigenvals,number_of_modes=5):
    min_extreme = -3
    pics_per_mode = 7

    fig_count = 1
    plt.figure()
    for i in range(number_of_modes):
        for j in range(pics_per_mode):
            plt.subplot(number_of_modes, pics_per_mode, fig_count)
            sdFromMean = (min_extreme+j) * np.sqrt(eigenvals[i])
            x_hat = sdFromMean * V[:,i] + mean
            reconstruction_x_hat = np.reshape(x_hat, [1, shape_y[1], shape_y[2]])
            plt.imshow(reconstruction_x_hat[0, :, :], cmap='gray')
            plt.axis('off')
            plt.title(str((min_extreme+j))+'sd')
            fig_count += 1

    plt.savefig('pictures/PCA_modesVariation_dim' + str(dimension) + '.png')
    plt.show()

PlotModesVaration()