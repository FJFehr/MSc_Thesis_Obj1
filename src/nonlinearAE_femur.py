# Running autoencoder pipeline for 3D meshes with non-linear activations
# Fabio Fehr
# 19 August 2020

from src.meshManipulation import loadMeshes, meshToData
from src.autoencoder_training import trainingAEViz, training_function
import glob2

def main():

    # Set the directory
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
    colour = [205, 155, 29]  # goldenrod3

    #### TRAINING ####

    # Full training scheme
    # param_grid = {'dimension': [50],
    #               'epochs': [20000],
    #               'learning_rate': [1e-4],
    #               'batch_size': [10],
    #               'regularization': [1e-4],
    #               'activation': ['selu','elu',"exponential",'hard_sigmoid']}

    # Best result
    param_grid = {'dimension': [50],
                  'epochs': [1000],
                  'learning_rate': [1e-4],
                  'batch_size': [10],
                  'regularization': [1e-5],
                  'activation': ["tanh"]}

    training_function(data, param_grid,name='femur_nonlinear_')

    #### VISUALISING ####

    # Set the directory and the wild cards to select all runs of choice
    direc = '../results/'
    paths = glob2.glob(direc + "*femur_nonlinear_tanh_AE_w2*")
    trainingAEViz(rawData, paths, triangles, "femur_nonlinear_AE_", colour, cameraName="femur",eigen_faust_Bool=False)

if __name__ == '__main__':
    main()