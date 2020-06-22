# This script performs pca on FAUST.
# Fabio Fehr
# 22 June 2020

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from meshManipulation import vecToMesh , \
    meshToVec, meshToData,mean3DVis,\
    loadMeshes,PlotModesVaration,\
    modesOfVariationVis,variationExplained,\
    PlotScatterGram,shapeParameters,meshVisSave


from sklearn.decomposition import PCA

def AnalyticalPCA(y, dimension):
    '''

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

def principalComponent3DVis(pca_obj, triangles, number_of_vis, name):
    '''
    This function takes in pca and saves 3D visualisations of the principal components
    :param pca_obj: the PCA object from sklearn.decomposition
    :param triangles: this shows the connections for the 3D object
    :param number_of_vis: How many PCs to visualise
    :param name: String name to be saved
    :return: nothing
    '''

    components = pca_obj.components_

    for i in range(0, number_of_vis):
        # create a mesh
        newMesh = vecToMesh(list=components[i, :],
                            triangles= triangles)

        meshVisSave(newMesh, "pictures/" + name + "PC" + str(i+1))



if __name__ == '__main__':

    # fetch data
    meshes = loadMeshes("meshes/")

    # create vertices dataset
    data = meshToData(meshes)

    # dimension to reduce
    dimension = 100

    # Now we have squeezed those 60000 down to 16 images of principal components
    pca_faust = AnalyticalPCA(data, dimension)  # has dimensions (16, 20670)

    # Get the components
    components = pca_faust.components_
    # Get the eigen values
    eigenvalues = pca_faust.singular_values_
    # Get the mean
    mean = pca_faust.mean_
    # Get triangles
    triangles = meshes[0].triangles

    # visualise and save the top 3 PCs
    principalComponent3DVis(pca_faust, triangles, 3, "faust_PCA_")

    #visualise and save the mean mesh
    mean3DVis(data, triangles,"faust_PCA_")

    # Get and save shape parameters
    b = shapeParameters(data, components)
    np.savetxt('faust_PCA_ShapeParamaters_b.csv', b, delimiter=',')

    # Save modes of variation
    #modesOfVariationVis(mean,components,eigenvalues,3,triangles,"faust_PCA_mov_")

    # Plot modes of variation
    #PlotModesVaration(3,"faust_PCA_",100)

    # Plot a basic scatterGram
    PlotScatterGram(b,4)

    # plot variation explained by PCA
    var_explained = variationExplained(eigenvalues)
    plt.plot(var_explained)
    plt.ylabel('explained_variance')
    plt.show()


    #TODO: Not sure why it crashes with (interrupted by signal 11: SIGSEGV) after PlotModesVaration