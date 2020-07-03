# This script reads in 3D meshes (FAUST) and trains a AE
# Fabio Fehr
# 22 June 2020

import open3d as o3d
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import glob2
from PIL import Image, ImageChops

# Some useful class types to keep in mind
# meshes[0]           is open3d.open3d_pybind.geometry.TriangleMesh
# meshes[0].vertices  are open3d.open3d_pybind.utility.Vector3dVector


def meshToVec(mesh):
    """
    This function takes in a mesh and returns a list of vertices
    This is important as the AE structure requires a vector and not a mesh

    :param mesh: open3d.open3d_pybind.geometry.TriangleMesh
    :return: a list of vertices
    """

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


def meshVisSave(mesh, path, col):
    '''
    This function saves a mesh visualisation as png

    :param mesh: 3D mesh obj from open3D
    :param path: path and name to where you would like it saved
    :param col: The colour in a list [255,255,255]
    :return: saves mesh
    '''
    # compute normals to visualise
    mesh.compute_vertex_normals()

    # convert the 255 code to be between 0-1 of Open3d
    col = [i / 255.0 for i in col]

    mesh.paint_uniform_color(col)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.capture_screen_image(path + ".png")
    vis.destroy_window()


def mean3DVis(data, triangles, name, col):
    '''
    Function takes in data and visualises the mean

    :param data: dataset rows are observations and columns are vertices (vector format)
    :param triangles: the connections for visualising meshes
    :param name: name to be saved
    :param col: The colour in a list [255,255,255]
    :return: nothing
    '''

    mean = data.mean(axis = 0)
    mean_mesh = vecToMesh(mean, triangles)
    meshVisSave(mean_mesh, "../results/" + name + "mean", col)\


def shapeParameters(data,components):

    '''
    This function takes in data and components from PCA or SVD and returns the shape paramters
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


def modesOfVariationVis(mean, components, singular_vals,number_of_modes,triangles, name, col):
    '''
    This function saves the individual modes of variation once they have been calculated

    :param mean: The mean shape
    :param components: the PC (PCA) or singular vectors (SVD)
    :param singular_vals: the eigenvals (PCA) or singular values (SVD)
    :param number_of_modes: how many modes do you want to visualise?
    :param triangles: The connection triangles for the mesh
    :param name: how will they be saved
    :param col: The colour in a list [255,255,255]
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
                meshVisSave(newMesh, '../results/' + name + "mode_" + str(i+1) + str((min_extreme+j)),col)


def trim(im):

    '''
    This function trims the white space of an image

    :param im: image
    :return: trimmed image
    '''
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def PlotModesVaration(number_of_modes, name):
    '''
    This function combines the modes of variation and plots them together and saves

    :param number_of_modes: how many modes do you want to plot
    :param name: the name of the saved file of pictures
    :return:
    '''

    min_extreme = -3
    pics_per_mode = 7

    fig_count = 1
    plt.figure()
    plt.tight_layout(pad=3)
    for i in range(number_of_modes):
        for j in range(pics_per_mode):
            plt.subplot(number_of_modes, pics_per_mode, fig_count)
            if (min_extreme+j == 0):
                img = Image.open('../results/'+ name + 'mean.png')
                img = trim(img)
            else:
                img = Image.open('../results/' + name + "mode_" + str(i+1) + str((min_extreme+j)) + '.png')
                img = trim(img)
            plt.imshow(img)
            plt.axis('off')
            if (min_extreme + j != 0):
                plt.title(f'${str((min_extreme+j))} \sqrt\lambda_{i+1}$', fontsize=10)
            else:
                plt.title("0", fontsize=10)
            fig_count += 1

    plt.savefig('../results/'+name+'modesVariation' + '.pdf', dpi=600)
    #plt.show()


def PlotScatterGram(b, num_of_modes, name):
    '''
    Makes a grouped scattergram of shape parameters

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
    plt.savefig("../results/"+name + 'scatterGrams.png')
