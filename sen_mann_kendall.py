'''
Author: 张仕山
Date: 2023-12-04 17:23:14
LastEditors: 张仕山
LastEditTime: 2023-12-09 21:04:57
Description:  
FilePath: \sen_mann_kendall.py
'''
from lib.GDALWrap import array2raster2, readImgDataByDirPath, BlockingComputing, noData
from osgeo import gdal
from numpy import *
import numpy as np
import glob
import os
from os import path
# import pymannkendall as mk
from numba import njit
import numba
# from numba import prange
numba.config.NUMBA_DEFAULT_NUM_THREADS=8


def writetoTIF(filename, im_proj, im_geotrans, im_data):
    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

def get_im(pathfile):
    ds = gdal.Open(pathfile)
    band = ds.GetRasterBand(1)
    arr_im = band.ReadAsArray()
    arr_im[np.isnan(arr_im)] = 0  # 将nan值替换为0，以免影响后面的运算
    return arr_im, ds.GetProjection(), ds.GetGeoTransform()

@BlockingComputing(5)
def csen(v_arr):
    # sen = np.zeros(v_arr[0].shape)
    data = []
    for i in range(len(v_arr)):
        j = i + 1
        while j < len(v_arr):
            s = (v_arr[j] - v_arr[i]) / (j - i)
            data.append(s)
            j += 1
    med = np.median(data, axis=0)
    sen = med

    return sen

# def mann_kendall(v_arr):
#     mkdata = np.zeros(v_arr[0].shape)
#     v_arr = np.asarray(v_arr)
#     for h in range(v_arr[0].shape[0]):
#         for k in range(v_arr[0].shape[1]):
#             datalist =v_arr[:,h,k]
#             # for i in range(len(v_arr)):
#             #     datalist.append(v_arr[i][h][k])
#             zd = mk.original_test(datalist,alpha=0.05)
#             mkdata[h][k] = zd.slope
#     return mkdata

# @njit(fastmath=True,parallel=True)
@njit
def mann_kendall(v_arr):
    mk = np.zeros(v_arr[0].shape)
    for h in range(v_arr[0].shape[0]):
        for k in range(v_arr[0].shape[1]):
            s = 0
            unique_x = []
            tp=[]
            for i in range(len(v_arr)):
                j = i + 1
                if v_arr[i][h][k] not in unique_x:
                    unique_x.append(v_arr[i][h][k])
                    tp.append(1)
                else:
                    index = unique_x.index(v_arr[i][h][k])
                    tp[index] +=1
                while j < len(v_arr):
                    j_i = v_arr[j][h][k] - v_arr[i][h][k]
                    s += np.sign(j_i)
                    j += 1
            n = len(v_arr)
            tp = np.asarray(tp)
            g = len(unique_x)
            # print(g)
            # var_s = (n * (n - 1) * (2 * n + 5)) / 18
            if n == g:  # 如果不存在重复点
                var_s = (n * (n - 1) * (2 * n + 5)) / 18
            else:
                var_s = (n * (n - 1) * (2 * n + 5) - np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18
            z = 0
            if s > 0:
                z = (s - 1) / np.sqrt(var_s)
            if s == 0:
                z = 0
            if s < 0:
                z = (s + 1) /  np.sqrt(var_s)
            mk[h][k] = z
    return mk

def smk(data):
    sen_im = data[0]
    mk_im = data[1]
    result = np.zeros(sen_im.shape)
    sda_005 = sen_im >= 0.003
    sde_005 = abs(sen_im) < 0.003
    sx_005 = sen_im <= -0.003
    mkda_196 = abs(mk_im) > 1.96
    mkx_196 = abs(mk_im) <= 1.96
    result[sda_005 & mkda_196] = 1
    result[sda_005 & mkx_196] = 2
    result[sx_005 & mkda_196] = 3
    result[sx_005 & mkx_196] = 4
    result[sde_005] = 5
    return result

def new_input_data(path, outPath):
    e_arr = readImgDataByDirPath(inputPath, ".tif")
    sen = csen(e_arr, 3)
    mk = mann_kendall(e_arr)
    sm = smk([sen,mk])

    print("结果输出中...")
    # writetoTIF(outPath, pro, trans, sm)
    # mk = mann_kendall(v_arr)
    # writetoTIF(r"F:\youmo\data\fujian\结果\sen_mann_kendall\sen1.tif", pro, trans, sen)
    fileFullPath = os.path.join(path, os.listdir(path)[0])

    array2raster2(outPath, fileFullPath, sm, 5)
    print('栅格图像组变异系数计算完成')

def input_data(path, outPath):
    e_arr = []
    filepath = glob.glob(os.path.join(path, "*.tif"))
    # filepath = [r"F:\youmo\data\test\result\devmax-mga\mask\tend\sen.tif",r"F:\youmo\data\test\result\devmax-mga\mask\tend\mk.tif"]

    for file in filepath:
        print(file)
        arr_im, pro, trans = get_im(file)
        e_arr.append(arr_im)
    
    sen = csen(np.asarray(e_arr))
    mk = mann_kendall(np.asarray(e_arr))
    sm = smk([sen,mk])


    print("结果输出中...")
    # writetoTIF(outPath, pro, trans, sm)
    # mk = mann_kendall(v_arr)
    # writetoTIF(r"F:\youmo\data\fujian\结果\sen_mann_kendall\sen1.tif", pro, trans, sen)
    fileFullPath = os.path.join(path, os.listdir(path)[0])
    array2raster2(outPath, fileFullPath, sm)
    print('栅格图像组变异系数计算完成')

if __name__ == '__main__':
    inputPath = r"F:\project\python\li\data\downloadall2\Maize\TWSO\result\result"

    outputDirPath = path.join(inputPath, "result")
    if not path.exists(outputDirPath):
       os.mkdir(outputDirPath)
    outPath = path.join(outputDirPath, "Maize_TWSO_C3S-glob-agric_smk.tif")
    new_input_data(inputPath, outPath)
