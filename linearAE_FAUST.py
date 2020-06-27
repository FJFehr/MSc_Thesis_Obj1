#
#
#


import numpy as np
import time
from meshManipulation import meshToData,\
    loadMeshes,PlotModesVaration,\
    modesOfVariationVis, \
    PlotScatterGram,shapeParameters
from keras.layers import Input, Dense
from keras import regularizers, models, optimizers
from sklearn.model_selection import ParameterGrid
import glob2


# Linear Autoencoder
def LinearAE(y, dimension, learning_rate = 1e-4, regularization = 5e-4,batch_size=4, epochs=3):

    '''
    This creates the basic frameowrk for a single layer linear AE
    :param y: data
    :param dimension: bottleneck hidden layer dimension
    :param learning_rate: for the adam optimiser
    :param regularization: regularisation parameter
    :param epochs: how many times the model sees the data
    :return: The weights and biases, loss per epoch, time elapsed
    '''

    start_time = time.time()

    input = Input(shape=(y.shape[1],))
    encoded = Dense(dimension, activation='linear',
                    kernel_regularizer=regularizers.l2(regularization))(input)
    decoded = Dense(y.shape[1], activation='linear',
                    kernel_regularizer=regularizers.l2(regularization))(encoded)
    autoencoder = models.Model(input, decoded)
    autoencoder.compile(optimizer=optimizers.adam(lr=learning_rate), loss='mean_squared_error')
    history = autoencoder.fit(y, y, epochs=epochs, batch_size=batch_size, shuffle=True, verbose =1)

    (w1,b1,w2,b2)=autoencoder.get_weights()

    end_time = time.time() - start_time

    return (w1,b1,w2,b2,history.history['loss'],end_time)

def train_AE_save(data,
                  dimension,
                  epochs,
                  learning_rate,
                  regularization,
                  batch_size,
                  name):
    '''
    This function takes in all the parameters for a linear AE and saves
     decoder weights, loss per epoch and the time taken to train in seconds

    :param data: data
    :param dimension: bottleneck hidden layer dimension
    :param learning_rate: for the adam optimiser
    :param regularization: regularisation parameter
    :param epochs: how many times the model sees the data
    :return: The weights and biases
    :param name: saving name
    :return:
    '''

    # Run linear AE and return and save the weights
    (_, _, w2, _,loss,time) = LinearAE(y= data,
                             dimension=dimension,
                             epochs=epochs,
                             learning_rate=learning_rate,
                             batch_size =batch_size,
                             regularization=regularization)

    np.savetxt('results/' + name +
               '_AE_loss_dim_' + str(dimension) +
               "_reg_" + str(regularization) +
               '_epoch_' + str(epochs) +
               '_lr_' + str(learning_rate) +
               '_bs_' + str(batch_size) +
               '.csv',
               loss, delimiter=',')

    np.savetxt('results/' + name +
               '_AE_time_dim_' + str(dimension) +
               "_reg_" + str(regularization) +
               '_epoch_' + str(epochs) +
               '_lr_' + str(learning_rate) +
               '_bs_' + str(batch_size) +
               '.csv',
               time, delimiter=',')


    np.savetxt('results/' + name +
               '_AE_w2_dim_'+str(dimension) +
               "_reg_" + str(regularization) +
               '_epoch_' + str(epochs) +
               '_lr_' + str(learning_rate) +
               '_bs_' + str(batch_size) +
               '.csv',
               w2, delimiter=',')

def training_function(data, param_grid):
    '''

    Here is an example parameter grid
    param_grid = {'dimension': [100],
               'epochs': [10000],
               'learning_rate': [1e-4],
                'batch_size': [5],
                'regularization': [1e-4]}

    :param data: Your data
    :param param_grid: a dict of your parameters to run.
    :return:
    '''

    # param_grid = {'dimension': [100],  # maybe try 20
    #               'epochs': [10000],
    #               'learning_rate': [1e-4],  # maybe 1e-6
    #               'batch_size': [5],
    #               'regularization': [1e-4]}  # maybe 1e-2

    params = list(ParameterGrid(param_grid))
    # loop through all options
    for i in range(0, len(params)):

        # Record times
        start_time = time.time()

        #show parameters
        for key, value in params[i].items():
            print(key,value)

        # train and save the AE weights
        train_AE_save(data=data,
                      name="faust",
                      **params[i])

        print("completed " + str(i/len(params)))
        print("--- %s seconds ---" % (time.time() - start_time))

def trainingAEViz(data, paths,triangles):

    '''
    Once training has been run you will want to visualise the modes of variation and scattergrams

    :param data: your data
    :param paths: The paths to your training results
    :param triangles: The triangles for your mesh visualisations
    :return:
    '''

    #Calculate the mean
    mean = data.mean(axis=0)

    for path in paths:

        #load the path to training output
        w2= np.loadtxt(str(path), delimiter=',')

        # Now these are equivalent to the principal components. svd on decoder weights
        (p_linear_ae, singular_values, _) = np.linalg.svd(w2.T, full_matrices=False)

        # Get the components
        components = p_linear_ae.T # 100 x 20670

        # turn to shape parameters
        b = shapeParameters(data, components)
        np.savetxt('results/faust_AE_ShapeParamaters_b.csv', b, delimiter=',')

        #From path get the new name
        name = str(path[20:-4])

        # Load eigenvalues from PCA for bounds
        pca_eigen_values = np.loadtxt('results/faust_PCA_Eigen.csv', delimiter=',')

        # Save the modes of variations pictures
        modesOfVariationVis(mean, components, pca_eigen_values, 3, triangles, name)

        # Plot
        PlotScatterGram(b, 3, "results/faust_AE_")


if __name__ == '__main__':

    # fetch data
    meshes = loadMeshes("meshes/")

    # create vertices dataset
    data = meshToData(meshes)

    # Get triangles
    triangles = meshes[0].triangles

    #### TRAINING ####

    # This set of parameters was the best and took 2.5hrs to train

    param_grid = {'dimension': [100],  # maybe try 20
                  'epochs': [100000], # maybe less?
                  'learning_rate': [1e-4],  # maybe 1e-6
                  'batch_size': [25], # maybe 5
                  'regularization': [1e-4]}  # maybe 1e-2
    training_function(data, param_grid)

    #### VISUALISING ####

    # Set the directory and the wild cards to select all runs of choice

    direc = 'results/'
    paths = glob2.glob(direc + "*faust_AE_w2_dim_100_reg_0.0001_epoch_10000_lr_0.0001_bs_25*")
    trainingAEViz(data, paths, triangles)

    # Appears to be the best run
    #results/AE_results/faust_AE_w2_dim_100_reg_0.0001_epoch_10000_lr_0.0001_bs_25.csv #
