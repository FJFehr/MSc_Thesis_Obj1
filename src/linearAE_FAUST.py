# Running autoencoder pipeline for 3D meshes with linear activations
# Fabio Fehr
# 1 July 2020

from src.meshManipulation import loadMeshes, meshToData
from src.autoencoder_training import trainingAEViz, training_function
import glob2

def main():

    # fetch data
    meshes = loadMeshes("../meshes/")

    # create vertices dataset
    data = meshToData(meshes)

    # Get triangles
    triangles = meshes[0].triangles

    # Set colour
    colour = [141, 182, 205]  # light blue colour

    #### TRAINING ####

    # Full training scheme
    # param_grid = {'dimension': [20,100],
    #               'epochs': [100,1000,10000],
    #               'learning_rate': [1e-6, 1e-4, 1e-2],
    #               'batch_size': [5, 25],
    #               'regularization': [0, 1e-2, 1e-4],
    #               'activation': ['linear']}

    # Best result
    param_grid = {'dimension': [100],
                  'epochs': [10],
                  'learning_rate': [1e-4],
                  'batch_size': [25],
                  'regularization': [1e-4],
                  'activation': ['linear']}

    training_function(data, param_grid,name='faust')

    #### VISUALISING ####

    # Set the directory and the wild cards to select all runs of choice

    # results/AE_results/faust_AE_w2_dim_100_reg_0.0001_epoch_10000_lr_0.0001_bs_25.csv #
    # Appears to be the best run

    direc = '../results/'
    paths = glob2.glob(direc + "*faust_AE_w2_dim_100_reg_0.0001_epoch_10000_lr_0.0001_bs_25*")
    trainingAEViz(data, paths, triangles, "faust_AE_", colour)

if __name__ == '__main__':
    main()
