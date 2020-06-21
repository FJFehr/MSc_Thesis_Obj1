# This script reads in 3D meshes (FAUST) and trains a AE
# Fabio Fehr
# 10 June 2020

import open3d as o3d
import numpy as np
import os

######################################################################################################################
# Mesh to Vector, Vector to Mesh #####################################################################################
######################################################################################################################
# Some useful class types to keep in mind
# meshes[0]           is open3d.open3d_pybind.geometry.TriangleMesh
# meshes[0].vertices  are open3d.open3d_pybind.utility.Vector3dVector

def meshToVec(mesh):
    # This function takes in a mesh and returns a list of vertices
    # This is important as the AE structure requires a a vector not mesh

    output = np.concatenate(np.asarray(mesh.vertices)).ravel().tolist()

    return output

def vecToMesh(list, triangles, N=3):
    # This function takes in a single list of vertices, triangles of the original mesh and dimension
    # returns a mesh of open3d

    # returns a numpy array grouped correctly
    groupednpArray = np.asarray([list[n:n + N] for n in range(0, len(list), N)])

    # create a 3D vector
    newVerticies = o3d.open3d_pybind.utility.Vector3dVector(groupednpArray)

    # Create the new mesh from the list
    newMesh = o3d.geometry.TriangleMesh(newVerticies, triangles)
    return newMesh

if __name__ == "__main__":
    ######################################################################################################################
    # Load dataset of meshes #############################################################################################
    ######################################################################################################################

    direc = "meshes/"
    paths = [os.path.join(direc, i) for i in os.listdir(direc)]
    meshes = [o3d.io.read_triangle_mesh(path) for path in paths]  # this is a list of meshes

    ######################################################################################################################
    # Basic visualisation ################################################################################################
    ######################################################################################################################
    current_mesh = meshes[0]

    print("This is what the original mesh looks like")
    current_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([current_mesh])

    # Convert to vector
    meshVec = meshToVec(current_mesh)
    print(meshVec)

    # Convert back to mesh
    newMesh = vecToMesh(list=meshVec, triangles=current_mesh.triangles)

    print("This is what the new mesh looks like")
    newMesh.compute_vertex_normals()
    newMesh.paint_uniform_color([1, 0.706, 0])  # Change colour so its easier to see who is who in the zoo
    o3d.visualization.draw_geometries([newMesh])