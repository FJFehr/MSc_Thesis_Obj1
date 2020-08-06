# This script performs pca on femur data.
# Fabio Fehr
# 23 July 2020

import numpy as np
from src.meshManipulation import \
    meshToData,mean3DVis,\
    loadMeshes,PlotModesVaration,\
    modesOfVariationVis, PlotScatterGram,shapeParameters, variationExplainedPlot
from sklearn.decomposition import PCA

def main():

    # fetch data
    meshes = loadMeshes("../meshes/femurs/",ply_Bool=False) # dims 50 36390

    # create vertices dataset
    data = meshToData(meshes)

    # Get triangles
    triangles = meshes[0].triangles

    # dimension to reduce to
    dimension = 50

    # Set the colour
    colour = [180, 180, 180] # Grey

    # Get the mean
    mean = data.mean(axis=0)

    # visualise and save the mean mesh
    mean3DVis(data, triangles, "femur_PCA_", col=colour, x_rotation=-400, y_rotation=-800)

    # PCA - The data is too large to do a full PCA, thus we do an SVD - This is built in in the PCA code
    pca = PCA(n_components=dimension)
    pca_femur = pca._fit_full(data, dimension) # This approximation is better.

    # Get components
    # components = pca.fit(data).components_  # pca original but approximation is kak
    components = pca_femur[2] # this makes sense in the scattergram. Its the V in SVD

    # Get eigenvalues
    # (all of these options lead to the same answer which is the SINGULAR VALUES - NOT EIGEN)
    # singularVals = np.linalg.svd(data,full_matrices=False)[1]
    # singularVals =  pca.fit(data).singular_values_

    #The link below shows the relationship between singular values and eigen values. It seems to make sense.
    # https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca

    singular_vals = pca._fit_full(data, dimension)[1]
    eigenvalues = (singular_vals) ** 2 / np.sqrt(50 - 1)
    np.savetxt('../results/femur_PCA_Eigen.csv', eigenvalues, delimiter=',')

    # Get and save shape parameters
    b = shapeParameters(data, components)
    np.savetxt('../results/femur_PCA_ShapeParamaters_b.csv', b, delimiter=',')

    # Save modes of variation
    modesOfVariationVis(mean, components, eigenvalues, 3, triangles, "femur_PCA_", col=colour,x_rotation=-400,y_rotation=-800)
    #
    # # Plot modes of variation
    PlotModesVaration(3, "femur_PCA_")

    # Plot a basic scatterGram
    PlotScatterGram(b, 3, "femur_PCA_")

    # plot variation explained by PCA
    variationExplainedPlot(eigenvalues, "femur_PCA_")
    print("fin")

if __name__ == '__main__':
    main()