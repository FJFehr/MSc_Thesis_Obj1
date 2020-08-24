# Running autoencoder pipeline for 3D meshes with linear activations
# Fabio Fehr
# 6 August 2020

from src.meshManipulation import loadMeshes, meshToData
from src.autoencoder_training import trainingAEViz, training_function
import glob2

def main():

    import os
    os.chdir("/media/fabio/Storage/UCT/Thesis/Coding/MSc_Thesis_Obj1/src")
    # fetch data
    meshes = loadMeshes("../meshes/femurs/", ply_Bool=False)  # dims 50 36390

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
    # param_grid = {'dimension': [50],
    #               'epochs': [10000],
    #               'learning_rate': [1e-4],
    #               'batch_size': [5,10],
    #               'regularization': [1e-1,1e-2, 1e-3],
    #               'activation': ['linear']}

    # Best result
    param_grid = {'dimension': [50],
                  'epochs': [20000],
                  'learning_rate': [1e-4],
                  'batch_size': [10],
                  'regularization': [1e-4],
                  'activation': ['linear']}

    # training_function(data, param_grid,name='femur_linear_')

    #### VISUALISING ####

    # Set the directory and the wild cards to select all runs of choice

    # results/femurlinear_AE_w2_dim_50_reg_0.0001_epoch_20000_lr_0.0001_bs_10.csv #
    # Appears to be the best run

    direc = '../results/'
    # paths = glob2.glob(direc + "*femur_linear_linear_AE_w2*")
    paths = glob2.glob(direc + "*femur_linear_linear_AE_w2_dim_50_reg_0.0001_epoch_20000_lr_0.0001_bs_10.csv*")
    trainingAEViz(rawData, paths, triangles, "femur_linear_AE_", colour, x_rotation=-400, y_rotation=-800,eigen_faust_Bool=False)
    print("fin")

if __name__ == '__main__':
    main()