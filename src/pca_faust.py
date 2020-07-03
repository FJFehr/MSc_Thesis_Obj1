# This script performs pca on FAUST.
# Fabio Fehr
# 22 June 2020

import numpy as np
from src.meshManipulation import \
    meshToData,mean3DVis,\
    loadMeshes,PlotModesVaration,\
    modesOfVariationVis, PlotScatterGram,shapeParameters
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def AnalyticalPCA(y, dimension):
    '''
    This fits basic PCA

    :param y: The data to reduce
    :param dimension: dimension to reduce to
    :return: pca fit object
    '''
    pca = PCA(n_components=dimension)
    return pca.fit(y)

    #  components_ : array, shape (n_components, n_features)
    #   returns the  Principal axes in feature space, representing the directions of maximum variance in the data
    # explained_variance_ : array, shape (n_components,)
    #   The amount of variance explained by each of the selected components.
    # singular_values_ : array, shape (n_components,)
    #   The singular values corresponding to each of the selected components.
    # mean_ : array, shape (n_features,)


def variationExplainedPlot(singular_values,name):
    """
    Creates a cummulative variation explained for plot for PCA.

    :param singular_values: for PCA
    :return: cum_variance_explained: cummulative variation explained
    """

    # calculate cumulative variance
    total_val = sum(singular_values)
    variance_explained = singular_values/total_val
    cum_variance_explained = np.cumsum(variance_explained)

    plt.figure()
    plt.plot(cum_variance_explained)
    plt.ylabel("Variation Explained")
    plt.xlabel("Principal Components")
    plt.savefig("../results/" + name + 'VariationExplained.png')


def main():

    # fetch data
    meshes = loadMeshes("../meshes/")

    # create vertices dataset
    data = meshToData(meshes)

    # Get triangles
    triangles = meshes[0].triangles

    # dimension to reduce to
    dimension = 100

    # Set the colour
    colour = [180, 180, 180] # Grey

    # PCA
    pca_faust = AnalyticalPCA(data, dimension)

    # Get the components
    components = pca_faust.components_

    # Get the eigen values
    eigenvalues = pca_faust.singular_values_
    np.savetxt('../results/faust_PCA_Eigen.csv', eigenvalues, delimiter=',')

    # Get the mean
    mean = pca_faust.mean_

    # visualise and save the mean mesh
    mean3DVis(data, triangles, "faust_PCA_", col=colour)

    # Get and save shape parameters
    b = shapeParameters(data, components)
    np.savetxt('../results/faust_PCA_ShapeParamaters_b.csv', b, delimiter=',')

    # Save modes of variation
    modesOfVariationVis(mean, components, eigenvalues, 3, triangles, "faust_PCA_", col=colour)

    # Plot modes of variation
    PlotModesVaration(3, "faust_PCA_")

    # Plot a basic scatterGram
    PlotScatterGram(b, 3, "faust_PCA_")

    # plot variation explained by PCA
    variationExplainedPlot(eigenvalues, "faust_PCA_")

if __name__ == '__main__':
    main()