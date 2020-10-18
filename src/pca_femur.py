# This script performs pca on femur data.
# Fabio Fehr
# 23 July 2020

import numpy as np
from src.meshManipulation import \
    meshToData,mean3DVis,\
    loadMeshes,PlotModesVaration,\
    modesOfVariationVis, PlotScatterGram,shapeParameters, variationExplainedPlot,meshVis
from sklearn.decomposition import PCA

def AnalyticalPCA(y, dimension, type = "sklearn"):
    '''
    This fits basic PCA

    :param y: The data to reduce
    :param dimension: dimension to reduce to
    :param type: 'sklearn' - The sklearn function, 'SVD' - using SVD, 'eigen' - using eigen decomposition
    :return: (components, eigenvalues)
    '''

    if(type == "sklearn"):
        # PCA with function
        print("Using sklearn function")
        pca = PCA(n_components=dimension)
        pca_fit = pca.fit(y)
        return(pca_fit.components_,pca_fit.singular_values_**2/y.shape[1])

    if (type == "SVD"):
        #Solve with SVD
        print("Using SVD")
        (u_vectors, singular_values, v_vectors) = np.linalg.svd((1 / y.shape[1]) * y, full_matrices=False)
        return (v_vectors, singular_values**2*y.shape[1])

    if (type == "eigen"): # Seems to only work if we center and divide by stdev
        # Solve with Eigen trick XX^T
        print("Using Eigen decomposition")
        proxyCov = 1 / y.shape[1] * (y.dot(y.T))
        (w, v) = np.linalg.eig(proxyCov)
        # Now project into the correct space
        big_v = (y.T).dot(v)
        # Normalise
        big_v_norm = big_v / np.linalg.norm(big_v, axis=0)
        return (big_v_norm, w)

def main():
    import os
    os.chdir("/media/fabio/Storage/UCT/Thesis/Coding/MSc_Thesis_Obj1/src")
    # fetch data
    meshes = loadMeshes("../meshes/femurs/", ply_Bool=False)

    # create vertices dataset
    rawData = meshToData(meshes)
    mean = rawData.mean(axis=0)
    # sd = np.std(rawData, axis=0)

    # Centering
    data = (rawData - mean)
    # print(data.shape) # dims 50, 35982 # Faust is roughly 20000

    # Get triangles
    triangles = meshes[0].triangles

    # dimension to reduce to
    dimension = 50

    # Set the colour
    colour = [180, 180, 180] # Grey
    # meshVis(meshes[0], colour)
    # find the position you like and press "p" This will save the camera position
    # save as femurCameraSettings.json in src

    # PCA - The data is too large to do a full PCA, thus we do an SVD - This is built in in the PCA code
    # PCA
    # (components,eigenvalues) = AnalyticalPCA(data, dimension, "sklearn")
    # print("sklearn eigenvalues", eigenvalues)
    # print("sklearn components", components)
    # (components,eigenvalues) = AnalyticalPCA(data, dimension, "eigen")
    # print("eigen eigenvalues", eigenvalues)
    # print("eigen components", components)
    (components, eigenvalues) = AnalyticalPCA(data, dimension, "SVD")
    print("SVD eigenvalues", eigenvalues)
    print("SVD components", components)
    eigenvalues = np.sqrt(eigenvalues * data.shape[1])**2/data.shape[0]
    np.savetxt('../results/femur_PCA_Eigen.csv', eigenvalues, delimiter=',')

    # visualise and save the mean mesh
    mean3DVis(rawData, triangles, "femur_PCA_", col=colour, cameraName="femur")

    # Get and save shape parameters
    b = shapeParameters(data, components)
    np.savetxt('../results/femur_PCA_ShapeParamaters_b.csv', b, delimiter=',')

    # Save modes of variation
    modesOfVariationVis(mean, components, eigenvalues, 3, triangles, "femur_PCA_", col=colour, cameraName="femur")

    # Plot modes of variation
    PlotModesVaration(3, "femur_PCA_")

    # Plot a basic scatterGram
    PlotScatterGram(b, 3, "femur_PCA_")

    # plot variation explained by PCA
    variationExplainedPlot(eigenvalues, "femur_PCA_")
    print("fin")

if __name__ == '__main__':
    main()