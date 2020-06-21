#
#
#

import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
from meshManipulation import vecToMesh , meshToVec
from sklearn.decomposition import PCA

######################################################################################################################
# Load dataset of meshes #############################################################################################
######################################################################################################################

direc = "meshes/"
paths = [os.path.join(direc, i) for i in os.listdir(direc)]
meshes = [o3d.io.read_triangle_mesh(path) for path in paths]  # this is a list of meshes

data = np.empty((100,20670)) # 100 x 20670
for i in range(0, len(meshes)):
    current_mesh = meshes[i]
    meshVec = meshToVec(current_mesh)
    data[i,:] = meshVec

######################################################################################################################
# PCA function #######################################################################################################
######################################################################################################################

# Analytical PCA
def AnalyticalPCA(y, dimension):
    pca = PCA(n_components=dimension)
    return pca.fit(y)
    # loadings = pca.components_
    # return loadings

#  components_ : array, shape (n_components, n_features)
#   returns the  Principal axes in feature space, representing the directions of maximum variance in the data
# explained_variance_ : array, shape (n_components,)
#   The amount of variance explained by each of the selected components.
# singular_values_ : array, shape (n_components,)
#   The singular values corresponding to each of the selected components.
# mean_ : array, shape (n_features,)

######################################################################################################################
# FAUST example ######################################################################################################
######################################################################################################################

dimension = 8
# load the data
y = data

# 100 examples of images in y, each 20670
shape_y = y.shape

# Now we have squeezed those 60000 down to 16 images of principal components
pca_faust = AnalyticalPCA(y,dimension) # has dimensions (16, 20670)

components_faust = pca_faust.components_
# Convert back to mesh
newMesh = vecToMesh(list=components_faust[1,:], triangles=meshes[0].triangles)

newMesh.compute_vertex_normals()
newMesh.paint_uniform_color([1, 0.706, 0])  # Change colour so its easier to see who is who in the zoo
#o3d.visualization.draw_geometries([newMesh])

# # plot the principal components
# def PlotResults(p,dimension):
#     sqrt_dimension = int(np.ceil(np.sqrt(dimension)))
#     plt.figure()
#     for i in range(p.shape[0]):
#         plt.subplot(sqrt_dimension, sqrt_dimension, i + 1)
#         plt.imshow(p[i, :, :],cmap='gray')
#         plt.title(str(i + 1))
#         plt.axis('off')
#     plt.savefig('pictures/PCA_components_dim' + str(dimension) + '.png')
#     plt.show()
#
# PlotResults(p_analytical,dimension)

# Plot the mean shape

pcaMean = pca_faust.mean_
meshMean = vecToMesh(list=pcaMean, triangles=meshes[0].triangles)

meshMean.compute_vertex_normals()
meshMean.paint_uniform_color([1, 0.6, 0])  # Change colour so its easier to see who is who in the zoo
#o3d.visualization.draw_geometries([meshMean])

# plot the pca reconstruction for the first shape

# Check out https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com

# n = 100, p = 20670, k = 16
# X(n,p), V(p,k)
# reconstruction = PC scores x Eigen vectors transposed + Mean
# reconstruction = X V V^T + Mean
# dimension X and Mean : 1x784 , V dimension (16, 784)

mean = pcaMean  # (784,) meaning a vector
eigenvals = pca_faust.singular_values_

X = y                     # meaning (100, 20670)
V = pca_faust.components_.T     # (20670, 16)
b = np.dot(X, V)          # PC Scores or shape parameters 100 x 16

hat_x = np.dot(b, V.T) + mean

# first reconstruction

newMesh = vecToMesh(list=hat_x[1,:], triangles=meshes[0].triangles)

newMesh.compute_vertex_normals()
newMesh.paint_uniform_color([1, 1, 0])  # Change colour so its easier to see who is who in the zoo
#o3d.visualization.draw_geometries([newMesh])

######################################################################################################################
# Scattergram ########################################################################################################
######################################################################################################################

# This is important to see if there are any non-linear dependencies in the data.

plt.scatter(b[:,0], b[:,1], alpha=0.5, s=0.5)
plt.title('Scattergram of shape parameters')
plt.xlabel('b1')
plt.ylabel('b2')
plt.axhline(0,color = "r" )
plt.axvline(0,color = "r")
#unhelpful in this context Very much between the bounds
# plt.xlim(-3*np.sqrt(eigenvals[0]),3*np.sqrt(eigenvals[0]))
# plt.ylim(-3*np.sqrt(eigenvals[1]),3*np.sqrt(eigenvals[1]))
plt.show()

#plot scattergrams of shape parameters.
def PlotResultsB(b,num_of_modes = dimension):
    plt.figure()
    fig_count = 1
    for i in range(num_of_modes):
        for j in range(num_of_modes):
            plt.subplot(num_of_modes, num_of_modes, fig_count)
            plt.scatter(b[:,i], b[:,j], alpha=0.2, s=0.5)
            plt.axhline(0, color="r")
            plt.axvline(0, color="r")
            fig_count += 1
    plt.show()

PlotResultsB(b,4)

plt.plot(np.cumsum(pca_faust.explained_variance_)/ sum(pca_faust.explained_variance_))
plt.ylabel('explained_variance')
plt.show()


sdFromMean = -3 * np.sqrt(eigenvals[0]) #* np.ones((1, dimension))
x_hat = sdFromMean * V[:,0] + pcaMean

newMesh = vecToMesh(list=x_hat, triangles=meshes[0].triangles)
newMesh.compute_vertex_normals()
newMesh.paint_uniform_color([1, 0.706, 0])  # Change colour so its easier to see who is who in the zoo

vis = o3d.visualization.Visualizer()
vis.create_window()
ctr = vis.get_view_control()
vis.add_geometry(newMesh)
vis.update_geometry(newMesh)
vis.poll_events()
# param = o3d.io.read_pinhole_camera_parameters("camera_params.json")
# ctr.convert_from_pinhole_camera_parameters(param)
#param = vis.get_view_control().convert_to_pinhole_camera_parameters()
#o3d.io.write_pinhole_camera_parameters("camera_params.json", param)
vis.capture_screen_image("pictures/newmesh2.png")

