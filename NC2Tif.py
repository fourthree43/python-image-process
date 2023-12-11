'''
Author: 张仕山
Date: 2023-12-09 16:05:08
LastEditors: 张仕山
LastEditTime: 2023-12-09 21:48:04
Description:  
'''
import os
import re
from lib.GDALWrap import array2raster, BatchingProcessor
import netCDF4 as nc

import numpy as np
from os import path

inputPath = r"F:\project\python\li\data\downloadall2\Maize\DVS"

@BatchingProcessor(inputPath, ".nc")
def nc2tif(imgPath):
   # 读取一下基本信息
    nc_data_obj = nc.Dataset(imgPath)
    Lon = nc_data_obj.variables["lon"][:]
    Lat = nc_data_obj.variables["lat"][:]

    # 读取变量的时候，会自动根据scale factor对数值进行还原，但是Nodata的栅格会存储为-32768
    # 无论是日数据还是小时数居，变量名都是"AOT_L2_Mean"

    AOD_arr = np.asarray(nc_data_obj.variables.get(
        'DVS', nc_data_obj.variables.get('TWSO')))  # 将AOD数据读取为数组
    
    # AOD_arr = np.array([AOD_arr])

    # 这个循环将所有Nodata的值（即-32768）全部改为0
    # for i in range(len(AOD_arr)):
    #     for j in range(len(AOD_arr[0])):
    #         if AOD_arr[i][j] == -32768:
    #             AOD_arr[i][j] = 0.0

    # 影像的四秩
    LonMin, LatMax, LonMax, LatMin = [
        Lon.min(), Lat.max(), Lon.max(), Lat.min()]

    # 分辨率计算，其实可以写死，都是2401*2401
    N_Lat = len(Lat)
    N_Lon = len(Lon)
    Lon_Res = (LonMax - LonMin) / (float(N_Lon) - 1)
    Lat_Res = (LatMax - LatMin) / (float(N_Lat) - 1)

    dirPath = path.dirname(imgPath)
    outputDirPath = path.join(dirPath, "result")
    if not path.exists(outputDirPath):
       os.mkdir(outputDirPath)

    fileName = os.path.splitext(path.basename(imgPath))[0]

    newRasterfn = os.path.join(os.path.abspath(
        outputDirPath), "{0}.tif".format(fileName))
    
    AOD_arr = AOD_arr.reshape((N_Lat, N_Lon ))
    AOD_arr = AOD_arr[::-1, :] # 上下翻转
    array2raster(newRasterfn, (LonMin, LatMax), Lon_Res,
                 Lat_Res, 4326, AOD_arr)

if __name__ == "__main__":
  print("开始处理...")
  nc2tif()
  print("处理完成！")