'''
Author: LIYALI
Date: 2023-05-19 17:24:05
LastEditors: LIYALI
LastEditTime: 2023-06-03 22:13:08
Description:  
FilePath: \test.py
'''
#!/usr/bin/env python
# --*-- encoding: utf-8 --*--

from osgeo import gdal, ogr, osr
import numpy as np
import struct
import os
import earthpy_plot_revised as ep
from datetime import datetime
import math

os.environ['PROJ_LIB'] = r'D:\Program Files (x86)\anaconda\pkgs\proj-6.2.1-h3758d61_0\Library\share\proj'


# gdal.GDT_Float32
def readRaster(input):
    # driver = gdal.GetDriverByName('GTiff')
    # driver.Open(input)
    dataSource = gdal.Open(input)
    band = dataSource.GetRasterBand(1)
    # 读取栅格数据
    # array = band.ReadRaster(xoff=0, yoff=0, xsize=band.XSize, ysize=band.YSize,
    #                             buf_xsize=band.XSize, buf_ysize=band.YSize, buf_type=type)
    # 读取栅格数据
    imgWidth = dataSource.RasterXSize
    imgHeight = dataSource.RasterYSize
    imgData = band.ReadAsArray(0, 0, imgWidth, imgHeight)
    return imgData

def readNCRaster(input):
    # driver = gdal.GetDriverByName('GTiff')
    # driver.Open(input)
    dataSource = gdal.Open(input)
    band = dataSource.GetRasterBand(1)
    # 读取栅格数据
    # array = band.ReadRaster(xoff=0, yoff=0, xsize=band.XSize, ysize=band.YSize,
    #                             buf_xsize=band.XSize, buf_ysize=band.YSize, buf_type=type)
    # 读取栅格数据
    imgWidth = dataSource.RasterXSize
    imgHeight = dataSource.RasterYSize
    imgData = band.ReadAsArray(0, 0, imgWidth, imgHeight)
    return imgData

def array2raster2(newRasterfn, mirrorImg, array):
    
    # 获取源影像的变换和空间坐标系参考
    source = gdal.Open(mirrorImg)
    geoTransform = source.GetGeoTransform()
    spatialRef = source.GetSpatialRef()

    # 创建输出影像
    driver = gdal.GetDriverByName('GTiff')
    rows = array.shape[0]
    cols = array.shape[1]
    if os.path.exists(newRasterfn):
        os.remove(newRasterfn)

    outRaster = driver.Create(
        '' if newRasterfn is None else newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    # 设置输出影像的变换和空间坐标系参考
    outRaster.SetGeoTransform(geoTransform)
    outRaster.SetProjection(spatialRef.ExportToWkt())

    # 写入影像数据
    outband = outRaster.GetRasterBand(1)
    outband.SetNoDataValue(-10000)
    outband.WriteArray(array)

    outband.FlushCache()
    source = None

def array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, EPSG, array):
    rows = array.shape[0]
    cols = array.shape[1]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]

    if newRasterfn is None:
      driver = gdal.GetDriverByName("MEM")
    else:
      driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(
        '' if newRasterfn is None else newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform(
        (originX, pixelWidth, 0, originY, 0, -pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.SetNoDataValue(0)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(EPSG)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    return outRaster
 
def dataVisualization(raster_arr):

    # 将图像的背景值设置为nan
    raster_arr[raster_arr <= 0] = 'nan'
    # 忽略nan值求最大和最小值 nanmin nanmax
    min_DN = np.nanmin(raster_arr)
    max_DN = np.nanmax(raster_arr)
    plot_title = "数据可视化"
    output_file_path_im = "./result/a.jpg"
    # 栅格数据可视化
    ep.plot_bands(raster_arr,
                    title=plot_title,
                    title_set=[25, 'bold'],
                    cmap="seismic",
                    cols=3,
                    figsize=(12, 12),
                    extent=None,
                    cbar=True,
                    scale=False,
                    vmin=-16,
                    vmax=45,
                    ax=None,
                    alpha=1,
                    norm=None,
                    save_or_not=True,
                    save_path=output_file_path_im,
                    dpi_out=600,
                    bbox_inches_out="tight",
                    pad_inches_out=0.1,
                    text_or_not=True,
                    text_set=[0.75, 0.95, "T(°C)", 20, 'bold'],
                    colorbar_label_set=True,
                    label_size=20,
                    cbar_len=2,
                    )  

def day2Month(days, year, end=True):
    endTimeStamp = datetime.strptime(str(year), "%Y").timestamp() + days * 24 * 60 * 60
    endDT = datetime.fromtimestamp(endTimeStamp)
    # 如果时间超过当前月份中旬，则取当前月(6月16 => 6)
    # else: (6月14 => 5月)
    if end:
        if endDT.day >= 15:
            endMonth = endDT.month
        else:
            endMonth = endDT.month - 1
    else:
        if endDT.day >= 15:
            endMonth = endDT.month + 1
        else:
            endMonth = endDT.month
    return endMonth
def calculateDroughtIndex(datas):
    # No Drought	(-0.5, +∞)
    # Mild Drought	(-1, −0.5]
    # Moderate Drought	(-1.5, −1]
    # Severe Drought	(-2, −1.5]
    # Extreme Drought	[-2, -∞]
    # 以SPEI=-0.5为干旱发生的阈值，则可以以干旱持续时间（duration，D）、强度（Intensity，Ｓ）和干旱频率（frequency，F）三个维度来衡量
    # 干旱持续时间Ｄ是干旱指数SPEI低于干旱事件阈值（SPEI>-0.5）的连续月数,即D1+D2+D3，
    # 干旱强度是SPEI>-0.5的和的绝对值即，
    # 干旱频率是发生干旱的月数与总作物生长期月数的比值
    data = np.asarray(datas)
    idx = np.where(data < -0.5)

    D = 0
    prevIdx = None
    for i in idx[0]:
        if (prevIdx != None):
            if i - prevIdx == 1:
                D += 1
            else:
                D = 0
        prevIdx = i
    if len(idx[0]) >= 1:
        D += 1
        
    S = abs(np.sum(data[idx]))
    if len(datas) == 0:
       F =0
    else:
       F = idx[0].shape[0] / len(datas)
    return D, S, F

def main():
    inputPlantingRegion = r"E:\climatecropdata\allcrop2010\calresult\resample\wheatWTindex_1_Resample.tif"
    inputCropPlantDay = r"E:\climatecropdata\allcrop2010\cropcal\resample\wheatWTplantday_Resample.tif"
    inputCropHarvestDay = r"E:\climatecropdata\allcrop2010\cropcal\resample\wheatWTharvestday_Resample.tif"
    outPath = "./result/"
    # 读取种植区域
    limitRegion = readRaster(inputPlantingRegion)
    # 获取种植区域
    gt0Index = np.where(limitRegion > 0)
    # 读取物候
    plantDate = readRaster(inputCropPlantDay)
    plantCropDate = plantDate[gt0Index]
    harvestDate = readRaster(inputCropHarvestDay)
    harvestCropDate = harvestDate[gt0Index]

    resultCropD = np.zeros(harvestCropDate.shape, dtype = float, order = 'C')
    resultCropS = np.zeros(harvestCropDate.shape, dtype = float, order = 'C')
    resultCropF = np.zeros(harvestCropDate.shape, dtype = float, order = 'C')
    # (1)获取北半球
    northIdxs = np.where(plantCropDate < harvestCropDate)
    # 根据物候计算指标
    # SPEI
    # 遍历
    SPEIPath = r"D:\data\spei\spei06\spei06_2010mon"
    baseYear = 2010
    # txtFormat = "spei05year_2010_01.nc"
    txtFormat = "spei06year_{0}_{1}.nc"
    datasets = {}
    northShape = northIdxs[0].shape
    northCropD = np.zeros(northShape, dtype = float, order = 'C')
    northCropS = np.zeros(northShape, dtype = float, order = 'C')
    northCropF = np.zeros(northShape, dtype = float, order = 'C')
    n = 0
    for idx in northIdxs[0]:
        endDay = int(harvestCropDate[idx])
        endMonth = day2Month(endDay, baseYear)
        # endTimeStamp = datetime.strptime(str(baseYear), "%Y").timestamp() + endDay * 24 * 60 * 60
        # endDT = datetime.fromtimestamp(endTimeStamp)
        # # 如果时间超过当前月份中旬，则取当前月(6月16 => 6)
        # # else: (6月14 => 5月)
        # if endDT.day >= 15:
        #     endMonth = endDT.month
        # else:
        #     endMonth = endDT.month - 1

        beginDay = int(plantCropDate[idx])
        beginMonth = day2Month(beginDay, baseYear, False)
        # beginTimeStamp = datetime.strptime(str(baseYear), "%Y").timestamp() + beginDay * 24 * 60 * 60
        # beginDT = datetime.fromtimestamp(beginTimeStamp)
        # if beginDT.day <= 15:
        #     beginMonth = beginDT.month
        # else:
        #     beginMonth = beginDT.month + 1
        
        # 遍历读取对应物候内的SPEI
        bands = []
        for i in range(beginMonth, endMonth + 1):
            fileName = txtFormat.format(baseYear, str(i).rjust(2, '0'))
            fileFullPath = os.path.join(SPEIPath, fileName)
            if os.path.exists(fileFullPath):
                if datasets.get(fileFullPath) is None:
                    band = readNCRaster(fileFullPath)
                    datasets[fileFullPath] = band
                else:
                    band = datasets.get(fileFullPath)
                bands.append(band)
            else: 
                print("文件不存在：{0}".format(fileFullPath))
        # 获取对应位置索引
        rate = harvestDate.shape[0] / band.shape[0]
        x = math.ceil((gt0Index[0][idx]) / rate) - 1
        y = math.ceil((gt0Index[1][idx]) / rate) - 1
        datas = list(map(lambda arr: arr[x, y], bands))
        # 计算当前位置指标
        D, S, F = calculateDroughtIndex(datas)
        northCropD[n] = D
        northCropS[n] = S
        northCropF[n] = F
        n += 1

    resultCropD[northIdxs] = northCropD
    resultCropS[northIdxs] = northCropS
    resultCropF[northIdxs] = northCropF
    # (2)获取南半球
    southIdxs = np.where(plantCropDate > harvestCropDate)
    # 根据物候计算指标
    # SPEI
    # 遍历
    southShape = southIdxs[0].shape
    southCropD = np.zeros(southShape, dtype = float, order = 'C')
    southCropS = np.zeros(southShape, dtype = float, order = 'C')
    southCropF = np.zeros(southShape, dtype = float, order = 'C')
    n = 0
    for idx in southIdxs[0]:
        endDay = int(harvestCropDate[idx])
        endMonth = day2Month(endDay, baseYear)

        beginDay = int(plantCropDate[idx])
        beginMonth = day2Month(beginDay, baseYear, False)
        
        # 遍历读取对应物候内的SPEI
        afterBands = []
        for i in range(1, endMonth + 1):
            fileName = txtFormat.format(baseYear, str(i).rjust(2, '0'))
            fileFullPath = os.path.join(SPEIPath, fileName)
            if os.path.exists(fileFullPath):
                if datasets.get(fileFullPath) is None:
                    band = readNCRaster(fileFullPath)
                    datasets[fileFullPath] = band
                else:
                    band = datasets.get(fileFullPath)
                afterBands.append(band)
            else: 
                print("文件不存在：{0}".format(fileFullPath))
        # 获取对应位置索引
        rate = harvestDate.shape[0] / band.shape[0]
        x = math.ceil((gt0Index[0][idx] + 1) / rate) - 1
        y = math.ceil((gt0Index[1][idx] + 1) / rate) - 1
        afterDatas = list(map(lambda arr: arr[x, y], afterBands))
        
        beforeBands = []
        for i in range(beginMonth, 12 + 1):
            fileName = txtFormat.format(baseYear-1, str(i).rjust(2, '0'))
            fileFullPath = os.path.join(SPEIPath, fileName)
            if os.path.exists(fileFullPath):
                if datasets.get(fileFullPath) is None:
                    band = readNCRaster(fileFullPath)
                    datasets[fileFullPath] = band
                else:
                    band = datasets.get(fileFullPath)
                beforeBands.append(band)
            else: 
                print("文件不存在：{0}".format(fileFullPath))
        # 获取对应位置索引
        beforeDatas = list(map(lambda arr: arr[x, y], beforeBands))


        # 计算当前位置指标
        D, S, F = calculateDroughtIndex(beforeDatas + afterDatas)
        southCropD[n] = D
        southCropS[n] = S
        southCropF[n] = F
        n += 1

    resultCropD[southIdxs] = southCropD
    resultCropS[southIdxs] = southCropS
    resultCropF[southIdxs] = southCropF


    # # 根据物候计算指标
    # totalDay1 = 365-plantCropDate[southIdxs] + harvestCropDate[southIdxs]
    # resultCrop[southIdxs] = totalDay1
    # result[gt0Index] = resultCrop
    # 输出
    
    # 以SPEI=-0.5为干旱发生的阈值，则可以以干旱持续时间（duration，D）、强度（Intensity，Ｓ）和干旱频率（frequency，F）三个维度来衡量
    # 创建同尺寸数组用于存储计算的指标
    resultD = np.zeros(limitRegion.shape, dtype = float, order = 'C')
    resultS = np.zeros(limitRegion.shape, dtype = float, order = 'C')
    resultF = np.zeros(limitRegion.shape, dtype = float, order = 'C')
    resultD[gt0Index] = resultCropD
    resultS[gt0Index] = resultCropS
    resultF[gt0Index] = resultCropF
    array2raster2(os.path.join(outPath, 'SPEI06_Duration_wheatWT_2010.tif'), inputPlantingRegion, resultD)
    array2raster2(os.path.join(outPath, 'SPEI06_Intensity_wheatWT_2010.tif'), inputPlantingRegion, resultS)
    array2raster2(os.path.join(outPath, 'SPEI06_Frequency_wheatWT_2010.tif'), inputPlantingRegion, resultF)
    

if __name__ == '__main__':
    print("开始执行...")
    main()
    print("执行完成...")
