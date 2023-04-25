import iris
import iris_grib
import numpy as np
import pyvista as pv
from pyvista import examples
import os
import sys


localdir = "/Users/erose/cira_workspace/cira-vertical-cloud-products/hdf_practice/files/files_grib/"
remotedir = "/mnt/data3/erose/files_grib/gribs/"


def loadLatLongConusDomain():
    dset_ll = np.load(localdir + "cld_data/2022001_lat_long.npz", mmap_mode="r")
    lat_proj = dset_ll["arr_0"][0]
    lon_proj = dset_ll["arr_0"][1] - 360.0
    lat_min = np.min(lat_proj); lat_max = np.max(lat_proj)
    lon_min = np.min(lon_proj); lon_max = np.max(lon_proj)
    conusLatProj = np.arange(lat_min, lat_max, np.abs(lat_min - lat_max)/1059.0)
    conusLonProj = np.arange(lon_min, lon_max, np.abs(lon_min - lon_max)/1799.0)
    return conusLatProj, conusLonProj

def extractHRRRCubesForVis():
    '''
    To define parameters in greater detail later, when we have more information about the type of files that are going to be loaded and processed
    currently only one file is downloaded that is only available locally.

    This function is responsible for loading in data from a HRRR (.grib2) file that is needed to appropriately plot 3D cloud isosurfaces with pyvista.
    In general, you just need Cloud Mixing Ratio data and Cloud Ice Mixing-Ratio data, but in this setup we also extract surface temperature data, and wind data
    to display surface temp as well as windbarbs at the specified vertical level
    '''
    dset_cmr = iris_grib.load_cubes(localdir + "20220831hrrr.t23z.wrfprsf00.grib2")#,"cloud_mixing_ratio")#, ["Cloud mixing ratio","Cloud ice mixing-ratio","isobaric","x","y"])
    cubes = list(dset_cmr)
    conusLatProj, conusLonProj = loadLatLongConusDomain()
    orig_lat_proj = cubes[0].coord("projection_y_coordinate").points / 17633.33333333
    orig_lon_proj = cubes[0].coord("projection_x_coordinate").points / 14983.33333333
    print(conusLatProj[0],conusLatProj[-1]); print(np.shape(conusLatProj))
    print(conusLonProj[0],conusLonProj[-1]); print(np.shape(conusLonProj))
    cmr_cubes = []; icmr_cubes = []
    t_sfc_cubes = []; wind_cubes = ["u","v","w"]
    for cube in cubes:
        # print(cube)
        cube.coord("projection_y_coordinate").points = conusLatProj
        cube.coord("projection_x_coordinate").points = conusLonProj
        ax = str(cube.aux_coords)
        attr = str(cube.attributes["GRIB_PARAM"])
        if attr == "GRIB2:d000c001n022": #Cloud mixing ratio
            # print(cube.standard_name)
            cmr_cubes.append(cube)
        elif attr == "GRIB2:d000c001n082": #Cloud ice mixing-ratio
            # print("Cloud ice mixing-ratio")
            cube.units  = "kg kg-1"
            icmr_cubes.append(cube)
        elif attr  == "GRIB2:d000c000n000" and "height" in ax:
            t_sfc_cubes.append(cube)
        elif attr == "GRIB2:d000c002n008" and "<DimCoord: pressure / (Pa)  [50000.]>" in ax: #vertical velocity
            print("vertical velocity"); print(ax)
            wind_cubes[2] = cube
        elif attr == "GRIB2:d000c002n002" and "<DimCoord: pressure / (Pa)  [50000.]>" in ax: #u component of wind
            wind_cubes[0] = cube
        elif attr == "GRIB2:d000c002n003" and "<DimCoord: pressure / (Pa)  [50000.]>" in ax: #v component of wind
            wind_cubes[1] = cube
    print("Cloud mixing ratio cubes: ", np.shape(cmr_cubes))
    print("Cloud ice mixing-ratio cubes: ", np.shape(icmr_cubes))
    print("Temp sfc cubes ", np.shape(t_sfc_cubes))
    print("Wind cubes ", np.shape(wind_cubes))
    wind_cubes = iris.cube.CubeList(wind_cubes)

    return cmr_cubes, icmr_cubes, t_sfc_cubes, wind_cubes, orig_lat_proj, orig_lon_proj


def extractCloudCondensate(cmr_cubes, icmr_cubes):
    summed = np.zeros((40,1059,1799),dtype="float32")
    for cube in range(len(cmr_cubes)):
        try:
            sum_data = np.zeros((1059,1799),dtype="float32")
            sum_data += cmr_cubes[cube].data
            sum_data += icmr_cubes[cube].data
            summed[cube] = sum_data
        # print(cube.data)
        except Exception as e:
            print(e)
    total_cmr_cubes = cmr_cubes
    for lvl in range(len(summed)):
        total_cmr_cubes[lvl].data = summed[lvl]
    total_cmr_cubes = iris.cube.CubeList(total_cmr_cubes)
    cmr_single_src = total_cmr_cubes.merge_cube()
    # print(cmr_single_src.coord("projection_y_coordinate").points)
    # print(cmr_single_src.coord("projection_x_coordinate").points)
    print(cmr_single_src)

    return cmr_single_src


def mapCloudProfilesToCondensate(cloudMatrix):
    for lvl in range(np.shape(cloudMatrix)[0]):
        ensureTerrain = False if lvl > 15 else True
        cloudMatrix[lvl] = maskCloudTypeToCmr(cloudMatrix[lvl],ensureTerrain)
    return cloudMatrix


def maskCloudTypeToCondensate(cloudMatrix):
    finalCldMatrix = np.zeros(np.shape(cloudMatrix),dtype="float32")
    waterLow = np.where(cloudMatrix == 2, 1, 0); waterHi = np.where(cloudMatrix == 3, 1, 0)
    superLow = np.where(cloudMatrix == 4, 1, 0); superHi = np.where(cloudMatrix == 5, 1, 0)
    mixedLow = np.where(cloudMatrix == 6, 1, 0); mixedHi = np.where(cloudMatrix == 7, 1, 0)
    # print(cloudMatrix[cloudMatrix == 6])
    iceLow = np.where(cloudMatrix == 8, 1, 0); iceHi = np.where(cloudMatrix == 9, 1, 0)
    finalCldMatrix[waterLow == 1] = 0.00003; finalCldMatrix[waterHi == 1] = 0.00003
    finalCldMatrix[superLow == 1] = 0.00004; finalCldMatrix[superHi == 1] = 0.00004
    finalCldMatrix[mixedLow == 1] = 0.00005; finalCldMatrix[mixedHi == 1] = 0.00005
    finalCldMatrix[iceLow == 1] = 0.00006; finalCldMatrix[iceHi == 1] = 0.00006
    # print(finalCldMatrix[finalCldMatrix == 0.00005])
    print(finalCldMatrix)
    return finalCldMatrix


def maskCloudTypeToCmr(cloudMatrix):
    finalCldMatrix = np.zeros(np.shape(cloudMatrix),dtype="float32")
    water = np.where(cloudMatrix == 1, 0.00003, 0)
    super = np.where(cloudMatrix == 2, 0.00004, 0)
    mixed = np.where(cloudMatrix == 3, 0.00005, 0)
    ice = np.where(cloudMatrix == 4, 0.00006, 0)
    
    finalCldMatrix[water > 0] = 0.00003
    finalCldMatrix[super > 0] = 0.00004
    finalCldMatrix[mixed > 0] = 0.00005
    finalCldMatrix[ice > 0] = 0.00006
    # print(finalCldMatrix)
    return finalCldMatrix


def maskByTerrain(cloudMatrix):
    finalCldMatrix = np.where(cloudMatrix == 6, 0.00002, 0)
    return finalCldMatrix


def sphere_with_texture_map(radius=1.0, lat_resolution=180, lon_resolution=360):
    """Sphere with texture coordinates.

    Parameters
    ----------
    radius : float, default: 1.0
        Sphere radius.

    lat_resolution : int, default: 100
        Set the number of points in the latitude direction.

    lon_resolution : int, default: 100
        Set the number of points in the longitude direction.

    Returns
    -------
    pyvista.PolyData
        Sphere mesh with texture coordinates.

    """
    theta, phi = np.mgrid[0 : np.pi : lat_resolution * 1j, 0 : 2 * np.pi : lon_resolution * 1j]
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    sphere = pv.StructuredGrid(x, y, z)
    texture_coords = np.empty((sphere.n_points, 2))
    texture_coords[:, 0] = phi.ravel('F') / phi.max()
    texture_coords[:, 1] = theta[::-1, :].ravel('F') / theta.max()
    sphere.active_t_coords = texture_coords
    return sphere.extract_surface(pass_pointid=False, pass_cellid=False)


def getGlobeMesh():
    globe = sphere_with_texture_map(radius=6371200.0, lat_resolution=90, lon_resolution=90)
    globe.textures["surface"] = examples.load_globe_texture()
    return globe

cloud_types = ["0001","0010","0011","0100"] #water, supercooled, mixed, ice


def isPixelCloud(pixel):
    bin_rep = format(pixel, '#09b')
    # print(bin_rep)
    pixel_type = str(bin_rep)[4:-1]
    # print(pixel_type)
    return pixel_type in cloud_types