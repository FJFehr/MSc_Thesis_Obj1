# This script reads in 3D meshes (FAUST) and trains a AE
# Fabio Fehr
# 22 June 2020

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob2

# Some useful class types to keep in mind
# meshes[0]           is open3d.open3d_pybind.geometry.TriangleMesh
# meshes[0].vertices  are open3d.open3d_pybind.utility.Vector3dVector

def meshToVec(mesh):
    """

    :param mesh: open3d.open3d_pybind.geometry.TriangleMesh
    :return: a list of vertices
    """

    # This function takes in a mesh and returns a list of vertices
    # This is important as the AE structure requires a a vector not mesh

    output = np.concatenate(np.asarray(mesh.vertices)).ravel().tolist()

    return output

def vecToMesh(list, triangles, N=3):
    """
    This function takes in a single list of vertices, triangles of the original mesh and dimension
    returns a mesh of open3d

    :param list: well a vector
    :param triangles: the connections for each
    :param N: the dimension, but its always 3
    :return: open3d.open3d_pybind.geometry.TriangleMesh
    """

    # returns a numpy array grouped correctly
    groupednpArray = np.asarray([list[n:n + N] for n in range(0, len(list), N)])

    # create a 3D vector
    newVerticies = o3d.open3d_pybind.utility.Vector3dVector(groupednpArray)

    # Create the new mesh from the list
    newMesh = o3d.geometry.TriangleMesh(newVerticies, triangles)
    return newMesh

def loadMeshes(direc= "meshes/"):
    '''
    Provided a directory of .ply meshes. This function reads them in and returns a list of meshes
    :param direc: directory of meshes
    :return: List of meshes
    '''
    paths = glob2.glob(direc + "*.ply")
    paths = sorted(paths) # makes sure its in the correct order
    meshes = [o3d.io.read_triangle_mesh(path) for path in paths]

    return meshes


def meshToData(meshes):
    '''
    This function turns a list of meshes into a np array of mesh vertices
    :param meshes: list of meshes
    :return: data: individual meshes on the rows, vertices on the columns
    '''

    number_of_meshes = len(meshes)
    # number of vertices times 3 for 3D
    number_of_vertices = len(np.asarray(meshes[0].vertices))*3

    data = np.empty([number_of_meshes, number_of_vertices])  # 100 x 20670 for faust
    for i in range(0, number_of_meshes):
        current_mesh = meshes[i]
        meshVec = meshToVec(current_mesh)
        data[i, :] = meshVec

    return data


def meshVisSave(mesh, path):

    '''
    # this function saves a mesh visualisation as png
    :param mesh: 3D mesh obj from open3D
    :param path: path and name to where you would like it saved
    :return: saves mesh
    '''
    # compute normals to visualise
    mesh.compute_vertex_normals()

    # paint
    # newMesh.paint_uniform_color([1, 0.706, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.capture_screen_image(path + ".png")
    vis.run()
    vis.destroy_window()

def mean3DVis(data, triangles, name):
    '''
    # Function takes in meshes and visualises the mean
    :param data: dataset rows are observations and columns are vertices (vector format)
    :param triangles: the connections for visualising meshes
    :param name: name to be saved
    :return: nothing
    '''

    mean = data.mean(axis = 0)
    mean_mesh = vecToMesh(mean, triangles)
    meshVisSave(mean_mesh, "pictures/" + name + "mean")\

def shapeParameters(data,components):

    '''
    This function takes in data and components from PCA or SVD and returns the shape paramters
    Check out
    https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com

    dimensions
    X(n,p), V(p,k)

    reconstruction = shapeParametes . Eigen vectors transposed + Mean
    reconstruction = (X.V) . V^T + Mean

    :param data: observations on the rows
    :param components: PCs (PCA) or singular vectors (SVD)
    :return: b: shape parameters
    '''

    X = data
    V = components.T
    b = np.dot(X, V)

    return b

def modesOfVariationVis(mean, components, singular_vals,number_of_modes,triangles, name):
    '''

    :param mean: The mean shape
    :param components: the PC (PCA) or singular vectors (SVD)
    :param singular_vals: the eigenvals (PCA) or singular values (SVD)
    :param number_of_modes: how many modes do you want to visualise?
    :param triangles: The connection triangles for the mesh
    :param name: how will they be saved
    :return: saves meshes
    '''

    min_extreme = -3
    pics_per_mode = 7

    for i in range(number_of_modes):
        for j in range(pics_per_mode):
            sdFromMean = (min_extreme+j) * np.sqrt(singular_vals[i])
            x_hat = sdFromMean * components[i,:] + mean
            newMesh = vecToMesh(x_hat, triangles)

            if (min_extreme+j != 0):
                meshVisSave(newMesh, 'pictures/' + name + "mode_" + str(i+1) + str((min_extreme+j)))

def PlotModesVaration(number_of_modes,name, dimension):
    '''

    :param number_of_modes: how many modes do you want to plot
    :param name: the name of the saved file of pictures
    :param dimension: Just for saving purposes
    :return:
    '''

    min_extreme = -3
    pics_per_mode = 7

    fig_count = 1
    plt.figure()
    for i in range(number_of_modes):
        for j in range(pics_per_mode):
            plt.subplot(number_of_modes, pics_per_mode, fig_count)
            if (min_extreme+j == 0):
                img = imageio.imread('pictures/'+ name + 'mean.png')
            else:
                img = imageio.imread('pictures/' + name + "mov_mode_" + str(i+1) + str((min_extreme+j)) + '.png')
            plt.imshow(img)
            plt.axis('off')
            plt.title(str((min_extreme+j))+'sd')
            fig_count += 1

    plt.savefig('pictures/PCA_modesVariation_dim' + str(dimension) + '.png')
    plt.show()


#plot scattergrams of shape parameters.
def PlotScatterGram(b,num_of_modes):
    '''
    Makes a grouped scattergram
    :param b: shape parameters
    :param num_of_modes: how many to display
    :return:
    '''
    plt.figure()
    fig_count = 1
    for i in range(num_of_modes):
        for j in range(num_of_modes):
            plt.subplot(num_of_modes, num_of_modes, fig_count)
            plt.scatter(b[:,i], b[:,j], alpha=0.8, s=1)
            plt.axhline(0, color="r")
            plt.axvline(0, color="r")
            fig_count += 1
    plt.show()

def variationExplained(singular_values):
    """

    :param singular_values:
    :return:
    """
    # calculate cumulative variance
    total_val = sum(singular_values)
    variance_explained = singular_values/total_val
    cum_variance_explained = np.cumsum(variance_explained)
    return cum_variance_explained


# TODO: Figure out how to zoom and crop FAUST images
# ctr = vis.get_view_control()
# # param = o3d.io.read_pinhole_camera_parameters("camera_params.json")
# # ctr.convert_from_pinhole_camera_parameters(param)
# #param = vis.get_view_control().convert_to_pinhole_camera_parameters()
# #o3d.io.write_pinhole_camera_parameters("camera_params.json", param)
