import numpy as np
from osgeo import gdal
import os
from tqdm import tqdm
import re

from lib.GDALWrap import readRaster, array2raster2, array2raster

from numba import njit
import numba 
numba.config.NUMBA_DEFAULT_NUM_THREADS=8
"""
计算hurst指数
"""

def s(inputdata):
    # 输入numpy数组
    n = inputdata.shape[0]
    t = 0
    for i in np.arange(n):
        if i <= (n - 1):
            for j in np.arange(i + 1, n):
                if inputdata[j] > inputdata[i]:
                    t = t + 1
                elif inputdata[j] < inputdata[i]:
                    t = t - 1
                else:
                    t = t
    return t

def beta(inputdata):
    n = inputdata.shape[0]
    t = []
    for i in np.arange(n):
        if i <= (n - 1):
            for j in np.arange(i + 1, n):
                t.append((inputdata[j] - inputdata[i]) / ((j - i) * 1.0))
    return np.median(t)

# @njit(fastmath=False,parallel=True)
def Hurst(x):
    # x为numpy数组
    n = x.shape[0]
    t = np.zeros(n - 1)  # t为时间序列的差分
    for i in range(n - 1):
        t[i] = x[i + 1] - x[i]
    mt = np.zeros(n - 1)  # mt为均值序列,i为索引,i+1表示序列从1开始
    for i in range(n - 1):
        mt[i] = np.sum(t[0:i + 1]) / (i + 1)

    # Step3累积离差和极差,r为极差
    r = []
    for i in np.arange(1, n):  # i为tao
        cha = []
        for j in np.arange(1, i + 1):
            if i == 1:
                cha.append(t[j - 1] - mt[i - 1])
            if i > 1:
                if j == 1:
                    cha.append(t[j - 1] - mt[i - 1])
                if j > 1:
                    cha.append(cha[j - 2] + t[j - 1] - mt[i - 1])
        r.append(np.max(cha) - np.min(cha))
    s = []
    for i in np.arange(1, n):
        ss = []
        for j in np.arange(1, i + 1):
            ss.append((t[j - 1] - mt[i - 1]) ** 2)
        s.append(np.sqrt(np.sum(ss) / i))
    r = np.array(r)
    s = np.array(s)
    xdata = np.log(np.arange(2, n))
    ydata = np.log(r[1:] / s[1:])
    # 分母加个小数防止分母为0
    #ydata = np.log(r[1:] / (s[1:] + 0.0000001))

    h, b = np.polyfit(xdata, ydata, 1)
    return h

def ImageHurst(imgpath,  outtif):
    """
    计算影像的hurst指数
    :param imgpath: 影像1，多波段
    :param outtif: 输出结果路径
    :return: None
    """
    # 读取影像1的信息和数据
    ds1 = gdal.Open(imgpath)
    projinfo = ds1.GetProjection()
    geotransform = ds1.GetGeoTransform()
    rows = ds1.RasterYSize
    colmns = ds1.RasterXSize
    data1 = ds1.ReadAsArray()
    print(data1.shape)

    src_nodta = ds1.GetRasterBand(1).GetNoDataValue()

    # 创建输出图像
    format = "GTiff"
    driver = gdal.GetDriverByName(format)
    dst_ds = driver.Create(outtif, colmns, rows, 1,gdal.GDT_Float32)
    dst_ds.SetGeoTransform(geotransform)
    dst_ds.SetProjection(projinfo)

    # 删除对象
    ds1 = None

    # 开始计算相关系数

    band1 = data1[0]
    out = band1 * 0 - 2222
    for row in tqdm(range(rows)):
        for col in range(colmns):
            if src_nodta is None:
                x = data1[:, row, col]
                hindex  =  Hurst(x)
                out[row, col] = hindex
            else:
                if band1[row, col] != src_nodta:
                    x = data1[:, row, col]
                    hindex = Hurst(x)
                    out[row, col] = hindex
    # 写出图像
    dst_ds.GetRasterBand(1).WriteArray(out)

    # 设置nodata
    dst_ds.GetRasterBand(1).SetNoDataValue(-2222)
    dst_ds = None

yearRe = r"(\d{4})"
# 按年份排序
def sortByTime(text):
   m = re.match(yearRe, text)
   if m:
      return m.group(1)
   return ""

noData = -10000
def newImageHurst(inputPath,  outtif):
    fileList = [x for x in os.listdir(inputPath) if x.endswith(".tif")]
    # 文件排序和分组
    fileList = sorted(fileList, key=sortByTime, reverse=False) #年份升序排列
    bandsData = None
    fileFullPath = ""
    msgBar = tqdm(list(enumerate(fileList)))
    for idx, file in msgBar:
        fileFullPath = os.path.join(inputPath, file)

        msgBar.set_description(f'文件数据读取中（{file}）')

        band = readRaster(fileFullPath)

        if bandsData is None:
            bandsData = np.zeros((len(fileList), *band.shape), dtype=float)
        bandsData[idx, :, :] = band
    msgBar.close()

    # 开始计算Hurst指数...
    print("开始计算Hurst指数...")

    shape = bandsData.shape
    rows = shape[1]
    colmns = shape[2]
    bands = shape[0]
    # result = np.zeros((rows, colmns), dtype="f")
    result = np.full((rows, colmns), noData, dtype="f")

    for row in tqdm(range(rows)):
        for col in range(colmns):
            x = bandsData[:, row, col].reshape(bands)
            nanLen = x[np.isnan(x)].shape[0]
            if nanLen == 0:
              try:
                hindex = Hurst(x)
                result[row, col] = hindex
              except Exception as e:
                print(x)
    # 写出图像
    # 输出
    print("结果输出中...")
    array2raster2(outtif, fileFullPath, result)
    print("计算完成！")

if __name__ == '__main__':
    # x = np.array([1.59, 1.57, 1.56, 1.54, 1.52, 1.50, 1.47, 1.43, 1.41, 1.40, 1.39])
    # print(Hurst(x))

    inputPath = r"F:\project\python\li\data\modis"
    outPath = r"F:\project\python\li\result\modis\hurst_ndvi.tif"
    newImageHurst(inputPath,  outPath)
