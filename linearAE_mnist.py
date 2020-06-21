# This script works through a linear AE for the MNIST dataset
# Fabio Fehr
# 16 June 2020

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from keras.layers import Input, Dense
from keras import regularizers, models, optimizers
import matplotlib.pyplot as plt
from keras.datasets import mnist

######################################################################################################################
# linear AE function #################################################################################################
######################################################################################################################

# notice this function has:
# single layer, linear activations, a regularisor, adam optimiser with learning rate,
# uses mean squared error loss, how many epochs, batch size, and shuffling

# Linear Autoencoder
def LinearAE(y, dimension, learning_rate = 1e-4, regularization = 5e-4, epochs=3):
    input = Input(shape=(y.shape[1],))
    encoded = Dense(dimension, activation='linear',
                    kernel_regularizer=regularizers.l2(regularization))(input)
    decoded = Dense(y.shape[1], activation='linear',
                    kernel_regularizer=regularizers.l2(regularization))(encoded)
    autoencoder = models.Model(input, decoded)
    autoencoder.compile(optimizer=optimizers.adam(lr=learning_rate), loss='mean_squared_error')
    autoencoder.fit(y, y, epochs=epochs, batch_size=4, shuffle=True)
    (w1,b1,w2,b2)=autoencoder.get_weights()
    return (w1,b1,w2,b2)

######################################################################################################################
# MNIST example ######################################################################################################
######################################################################################################################

# In reconstruction the number of dimensions change the number of eigen components used. This creates the overall form
# of the handwritten number and then the epochs removes blurriness. The lines become more defined.

# set dimensions then just run the script
dimension = 16
epochs = 3

# load the data
(y, _), (_, _) = mnist.load_data()

# 60000 examples of images in y, each 28 by 28
shape_y = y.shape # (60000, 28, 28)

# reshape y to be a 2D matrix of the dataset
y = np.reshape(y,[shape_y[0],shape_y[1]*shape_y[2]]).astype('float32')/255 # (60000, 784)

# Now we have squeezed those 60000 down to a bottleneck and train the AE
(_, _, w2, _) = LinearAE(y, dimension, epochs=epochs)

# Now these are equivalent to the principal components. svd on decoder weights
(p_linear_ae, singular_values, _) = np.linalg.svd(w2.T, full_matrices=False)

# reshape before plotting
p_linear_ae = np.reshape(p_linear_ae.T, [dimension, shape_y[1], shape_y[2]]) # singular vectors U
w2 = np.reshape(w2,[dimension,shape_y[1],shape_y[2]]) # the weights

# plot the components
def PlotResults(p,dimension, name):
    sqrt_dimension = int(np.ceil(np.sqrt(dimension)))
    plt.figure()
    for i in range(p.shape[0]):
        plt.subplot(sqrt_dimension, sqrt_dimension, i + 1)
        plt.imshow(p[i, :, :],cmap='gray')
        plt.title(str(i+1))
        plt.axis('off')

    plt.savefig('pictures/AE_components_dim' + str(dimension) + '_epochs' + str(epochs) + '.png')
    #plt.show()

PlotResults(w2,dimension,'W2')
PlotResults(p_linear_ae, dimension, 'LinearAE_PCA')

# # Plot the mean digit
mnist_mean = y.mean(axis =0) #calculate row means
mnist_mean_plot = np.reshape(mnist_mean,[1,shape_y[1],shape_y[2]]) # has dimensions (1, 28, 28)

plt.figure()
plt.title("The mean image")
plt.imshow(mnist_mean_plot[0, :, :], cmap='gray')
plt.axis('off')
plt.show()

# plot the first image (Its a 5)

firstImage =  np.reshape(y[0, :],[1,shape_y[1],shape_y[2]])
plt.figure()
plt.title("First image = 5")
plt.imshow(firstImage[0, :, :], cmap='gray')
plt.axis('off')
plt.show()

# plot the AE reconstruction for the first image

# Check out https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com

# n = 60000, p = 784, k =16
# X(n,p), V(p,k)
# reconstruction = PC scores x Eigen vectors transposed + Mean
# reconstruction = X V V^T + Mean
# dimension X and Mean : 1x784 , V dimension (16, 784)

X = y                     # meaning (60000, 784)
V = np.reshape(p_linear_ae.T,[shape_y[1]*shape_y[2],dimension])     # (784, 16)
b = np.dot(X, V)          # shape parameters 60000 x 16

hat_x = np.dot(b, V.T) + mnist_mean

reconstructionImage = np.reshape(hat_x, [60000, shape_y[1], shape_y[2]])
plt.figure()
plt.title("Reconstruction")
plt.imshow(reconstructionImage[0, :, :], cmap='gray')
plt.axis('off')
plt.savefig('pictures/AE_dim' + str(dimension) + '_epochs' + str(epochs) + '.png')
plt.show()

######################################################################################################################
# Scattergram ########################################################################################################
######################################################################################################################

# This is important to see if there are any non-linear dependencies in the data.
# we plot the shape parameters b and look for relationships

# now lets plot the shape parameters against one another.
plt.scatter(b[:,0], b[:,1], alpha=0.5, s=0.5)
plt.title('Scattergram of shape parameters')
plt.xlabel('b1')
plt.ylabel('b2')
plt.axhline(0,color = "r" )
plt.axvline(0,color = "r")
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
            plt.axhline(0, color="r")
            plt.axvline(0, color="r")
            fig_count += 1
    plt.show()

PlotResultsB(b,4)

######################################################################################################################
# Modes of variation  ##########################################################################################
######################################################################################################################

# Here we will plot the modes of variation at their extremes -3sqrt(lamba) and 3sqrt(lambda)

def PlotModesVaration(mean=mnist_mean,sing_vals=singular_values,number_of_modes=5):
    min_extreme = -3
    pics_per_mode = 7

    fig_count = 1
    plt.figure()
    for i in range(number_of_modes):
        for j in range(pics_per_mode):
            plt.subplot(number_of_modes, pics_per_mode, fig_count)
            sdFromMean = (min_extreme+j) * np.sqrt(sing_vals[i]) #* np.ones((1, dimension))
            x_hat = sdFromMean * V[:,i] + mean
            reconstruction_x_hat = np.reshape(x_hat, [1, shape_y[1], shape_y[2]])
            plt.imshow(reconstruction_x_hat[0, :, :], cmap='gray')
            plt.axis('off')
            plt.title(str((min_extreme+j))+'sd')
            fig_count += 1

    plt.savefig('pictures/AE_modesVariation_dim' + str(dimension) + '_epochs' + str(epochs) + '.png')
    plt.show()

PlotModesVaration()