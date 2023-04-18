import iris
import iris_grib
import numpy as np
import os
import sys


localdir = "/Users/erose/cira_workspace/cira-vertical-cloud-products/hdf_practice/files/files_grib/"
remotedir = "/mnt/data3/erose/files_grib/gribs/"


def loadHRRRData():
    '''
    To define parameters in greater detail later, when we have more information about the type of files that are going to be loaded and processed
    currently only one file is downloaded that is only available locally.
    '''
    dset_ll = np.load(localdir + "cld_data/2022001_lat_long.npz", mmap_mode="r")
    dset_cmr = iris_grib.load_cubes(localdir + "20220831hrrr.t23z.wrfprsf00.grib2")#,"cloud_mixing_ratio")#, ["Cloud mixing ratio","Cloud ice mixing-ratio","isobaric","x","y"])
    cubes = list(dset_cmr)
    lat_proj = dset_ll["arr_0"][0]
    lon_proj = dset_ll["arr_0"][1] - 360.0
    lat_min = np.min(lat_proj); lat_max = np.max(lat_proj)
    lon_min = np.min(lon_proj); lon_max = np.max(lon_proj)
    finalLatProj = np.arange(lat_min, lat_max, np.abs(lat_min - lat_max)/1059.0)
    finalLonProj = np.arange(lon_min, lon_max, np.abs(lon_min - lon_max)/1799.0)
    orig_lat_proj = cubes[0].coord("projection_y_coordinate").points / 17633.33333333
    orig_lon_proj = cubes[0].coord("projection_x_coordinate").points / 14983.33333333
    print(finalLatProj[0],finalLatProj[-1]); print(np.shape(finalLatProj))
    print(finalLonProj[0],finalLonProj[-1]); print(np.shape(finalLonProj))
    cmr_cubes = []; icmr_cubes = []
    t_sfc_cubes = []
    for cube in cubes:
        # print(cube)
        cube.coord("projection_y_coordinate").points = finalLatProj #finalLatProj
        cube.coord("projection_x_coordinate").points = finalLonProj #finalLonProj
        ax = str(cube.aux_coords)
        attr = str(cube.attributes["GRIB_PARAM"])
        if attr == "GRIB2:d000c001n022":
            # print(cube.standard_name)
            cmr_cubes.append(cube)
        elif attr == "GRIB2:d000c001n082":
            # print("Cloud ice mixing-ratio")
            cube.units  = "kg kg-1"
            icmr_cubes.append(cube)
        elif attr  == "GRIB2:d000c000n000" and "height" in ax:
            t_sfc_cubes.append(cube)
        # print(cube.coord("projection_y_coordinate").points)
        # print(cube.coord("projection_x_coordinate").points)
        # print(cube.coord_system())

    print("Cloud mixing ratio cubes: ", np.shape(cmr_cubes))
    print("Cloud ice mixing-ratio cubes: ", np.shape(icmr_cubes))  
    print("Temp sfc cubes ", np.shape(t_sfc_cubes))

    return cmr_cubes, icmr_cubes, t_sfc_cubes, orig_lat_proj, orig_lon_proj


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
    print(cmr_single_src.coord("projection_y_coordinate").points)
    print(cmr_single_src.coord("projection_x_coordinate").points)
    print(cmr_single_src)

    return cmr_single_src