# Running autoencoder pipeline for 3D meshes with linear activations
# Fabio Fehr
# 1 July 2020

from src.meshManipulation import loadMeshes, meshToData
from src.autoencoder_training import trainingAEViz, training_function
import glob2

def main():

    # fetch data
    meshes = loadMeshes("../meshes/faust/")

    # create vertices dataset
    rawData = meshToData(meshes)
    mean = rawData.mean(axis=0)

    # center the data
    data = (rawData - mean)

    # Get triangles
    triangles = meshes[0].triangles

    # Set colour
    colour = [141, 182, 205]  # light blue colour

    #### TRAINING ####

    # Full training scheme
    # param_grid = {'dimension': [20,100],
    #               'epochs': [100,1000,10000,20000],
    #               'learning_rate': [1e-6, 1e-4, 1e-2],
    #               'batch_size': [5, 25],
    #               'regularization': [0, 1e-2, 1e-4],
    #               'activation': ['linear']}

    # Best result
    param_grid = {'dimension': [100],
                  'epochs': [1000],
                  'learning_rate': [1e-4],
                  'batch_size': [25],
                  'regularization': [1e-5],
                  'activation': ['linear']}

    # training_function(data, param_grid,name='faust_linear_')

    #### VISUALISING ####

    # Set the directory and the wild cards to select all runs of choice

    direc = '../results/'
    # paths = glob2.glob(direc + "*faust_linear_linear_AE_w2_*")
    paths = glob2.glob(direc + "faust_linear_linear_AE_w2_*")
    trainingAEViz(rawData, paths, triangles, "faust_linear_AE_", colour, cameraName="faust",
                  eigen_faust_Bool=True,trim_type="faust")

if __name__ == '__main__':
    main()
