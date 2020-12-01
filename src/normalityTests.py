# This scripts tests the normality of
# 1. Femur mesh vertices
# 2. FAUST mesh vertices
# 3. Femur shape parameters
# 4. FAUST shape parameters

# 30 November 2020

from scipy.stats import shapiro,normaltest,jarque_bera
from statsmodels.stats.diagnostic import lilliefors
from src.meshManipulation import loadMeshes, meshToData
import os
from pingouin import multivariate_normality
import pandas as pd
import numpy as np

cutoff = 0.05

# fetch data
os.chdir("/media/fabio/Storage/UCT/Thesis/Coding/MSc_Thesis_Obj1/src")
# fetch data
faustmeshes = loadMeshes("../meshes/faust/")
femurmeshes = loadMeshes("../meshes/femurs/", ply_Bool=False)

faustShapeParameters = np.genfromtxt("../results/faust_PCA_ShapeParamaters_b.csv",delimiter=",")
femurShapeParameters = np.genfromtxt("../results/femur_PCA_ShapeParamaters_b.csv",delimiter=",")

# create vertices dataset
faustData = meshToData(faustmeshes)
femurData = meshToData(femurmeshes)


def normalityTestCoordiate(data, cutoff):
    """
    This tests the univariate normality of the x , y , z of a mesh's vertices individually

    Tests are Shapiro Wilks, Jarque Bera, D Agostino, Lilliefors

    :param data:
    :param cutoff:
    :return: proportion that pass all tests
    """
    shapiroList = []
    jarque_beraList = []
    D_AgostinoList = []
    LillieforsList = []
    totalList =[]

    print("Running univariate normality tests on individual x,y,z")
    for i in range(data.shape[1]):

        # Select the variable
        currentvariable = data[:, i]

        # Perform tests individual tests
        _, shapiro_p = shapiro(currentvariable)
        if (shapiro_p >= cutoff):
            shapiro_count = 1
        else:
            shapiro_count = 0
        shapiroList.append(shapiro_count)

        _, jarque_bera_p = jarque_bera(currentvariable)
        if (jarque_bera_p >= cutoff):
            jarque_bera_count = 1
        else:
            jarque_bera_count = 0
        jarque_beraList.append(jarque_bera_count)

        _, D_Agostino_p = normaltest(currentvariable)
        if (D_Agostino_p >= cutoff):
            D_Agostino_count = 1
        else:
            D_Agostino_count = 0
        D_AgostinoList.append(D_Agostino_count)

        _, lilliefors_p = lilliefors(currentvariable)
        if (lilliefors_p >= cutoff):
            lilliefors_count = 1
        else:
            lilliefors_count = 0
        LillieforsList.append(lilliefors_count)

        if(shapiro_count == jarque_bera_count ==D_Agostino_count == lilliefors_count ==1):
            totalList.append(1)
        else:
            totalList.append(0)


    # format into dict
    coordiate_dict = {'shapiro': shapiroList,
                      'jarque_bera': jarque_beraList,
                      'D_agostino': D_AgostinoList,
                      'Lilliefors': LillieforsList}

    # make dataframe
    coordiate_df = pd.DataFrame(coordiate_dict)

    # get frequencies
    counts_df = pd.Index(coordiate_df.sum(axis=1)).value_counts(normalize=True)

    # return proportion of x,y,z points that pass tests
    # print(counts_df)
    return sum(totalList)/len(totalList)


def normalityTest3Dpoints(data, cutoff):
    """
    This tests the multivariate normality of the (x , y , z) 3D points of a mesh's vertices

    Test is Henze-Zirkler Normality Test
    :param data:
    :param cutoff:
    :return: the proportion of points that pass the test
    """

    print("Running multivariate normality tests on 3D points (x,y,z)")
    # considering the (x, y, z) point as a multivariate normal variable
    # How many 3D points are normally distributed?
    HZList = []
    for i in range(int(data.shape[1] / 3)):
        # for i in range(10):
        currentVariables = data[:, i * 3:i * 3 + 3]

        HZresults = multivariate_normality(currentVariables, alpha=cutoff)

        if (HZresults.normal):
            HZ_count = 1
        else:
            HZ_count = 0
        HZList.append(HZ_count)

    # how many points pass have 1s?
    return sum(HZList) / len(HZList)



def normalityTestS(data, cutoff):

    """
    This tests the entire S vector with all points for multivariate normality

    :param data:
    :param cutoff:
    :return: p-value of multivariate normality
    """
    print("Running full multivariate normality test")
    HZresults = multivariate_normality(data, alpha=cutoff)
    return HZresults.pval

# how many coordiates pass and have all 1s?

# POINTS
print("FAUST")
print(normalityTest3Dpoints(faustData,cutoff))
print(normalityTestCoordiate(faustData, cutoff))


print("Femur")
print(normalityTest3Dpoints(femurData,cutoff))
print(normalityTestCoordiate(femurData,cutoff))

#Full normallity of s? THESE BLOW UP
# print(normalityTestS(femurData,cutoff))
# print(normalityTestS(faustData, cutoff))

# SHAPE PARAMETERS
print("FAUST")
print(normalityTestCoordiate(faustShapeParameters[:,0:2], cutoff))
print(normalityTestS(faustShapeParameters[:,0:2], cutoff))

print("Femur")
print(normalityTestCoordiate(femurShapeParameters[:,0:2],cutoff))
print(normalityTestS(femurShapeParameters[:,0:2], cutoff))

# find points that are non-normal on faust and femur and colour them red?
print("fin")