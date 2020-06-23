#
#
#

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from meshManipulation import vecToMesh , \
    meshToVec, meshToData,mean3DVis,\
    loadMeshes,PlotModesVaration,\
    modesOfVariationVis,variationExplained,\
    PlotScatterGram,shapeParameters,meshVisSave
from keras.layers import Input, Dense
from keras import regularizers, models, optimizers



# Linear Autoencoder
def LinearAE(y, dimension, learning_rate = 1e-4, regularization = 5e-4, epochs=3):

    '''
    This creates the basic frameowrk for a single layer linear AE
    :param y: data
    :param dimension: bottleneck hidden layer dimension
    :param learning_rate: for the adam optimiser
    :param regularization: regularisation parameter
    :param epochs: how many times the model sees the data
    :return: The weights and biases
    '''

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

if __name__ == '__main__':

    # dimension to reduce
    dimension = 100
    epochs = 10

    # fetch data
    meshes = loadMeshes("meshes/")

    # create vertices dataset
    data = meshToData(meshes)


    # Now we have squeezed those 20670 down to a bottleneck and train the AE
    #(_, _, w2, _) = LinearAE(data, dimension, epochs=epochs)
    # save weights of decoder
    #np.savetxt('results/faust_AE_w2_dim_'+ str(dimension) + '_epoch_' + str(epochs) +'.csv', w2, delimiter=',')
    # read weights of decoder
    w2 = np.loadtxt('results/faust_AE_w2_dim_'+ str(dimension) + '_epoch_'+str(epochs)+'.csv', delimiter=',')
    # Now these are equivalent to the principal components. svd on decoder weights
    (p_linear_ae, singular_values, _) = np.linalg.svd(w2.T, full_matrices=False)

    # Get the components
    components = p_linear_ae.T # 100 x 20670

    # Get triangles
    triangles = meshes[0].triangles

    # np.savetxt('results/faust_AE_SingularVals.csv', singular_values, delimiter=',')
    # # Get the mean
    mean = data.mean(axis = 0)
    #mean3DVis(data, triangles, "faust_AE_")

    # Get and save shape parameters
    b = shapeParameters(data, components)
    np.savetxt('results/faust_AE_ShapeParamaters_b.csv', b, delimiter=',')

    # Save modes of variation
    #modesOfVariationVis(mean,components,singular_values,3,triangles,"faust_AE_mov_")

    # Plot modes of variation
    #PlotModesVaration(3,"faust_AE_",dimension)

    # Plot a basic scatterGram
    PlotScatterGram(b, 4)

