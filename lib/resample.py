'''
Created on 2020年2月10日

@author: Sun Strong
'''
from osgeo import gdal
import os
import numpy as np


def ReprojectImages2(inputfilePath,referencefilefilePath,output):
    # 若采用gdal.Warp()方法进行重采样
    # 获取输出影像信息
    inputrasfile = gdal.Open(inputfilePath, gdal.GA_ReadOnly)
    inputProj = inputrasfile.GetProjection()
    # 获取参考影像信息
    referencefile = gdal.Open(referencefilefilePath, gdal.GA_ReadOnly)
    referencefileProj = referencefile.GetProjection()
    referencefileTrans = referencefile.GetGeoTransform()
    bandreferencefile = referencefile.GetRasterBand(1)
    x = referencefile.RasterXSize
    y = referencefile.RasterYSize
    nbands = referencefile.RasterCount
    # 创建重采样输出文件（设置投影及六参数）
    driver = gdal.GetDriverByName('GTiff')
    output = driver.Create(outputfilePath, x, y, nbands, bandreferencefile.DataType)
    output.SetGeoTransform(referencefileTrans)
    output.SetProjection(referencefileProj)
    options = gdal.WarpOptions(srcSRS=inputProj, dstSRS=referencefileProj, resampleAlg=gdalconst.GRA_Bilinear)
    gdal.Warp(output, inputfilePath, options=options)

def main():
    # 批处理
    dirPath = r"E:\testdata\riqimazie2010"
    outputDirPath = r"E:\testdata\riqimazie2010TIFF"
    referencefilefilePath='E:\testdata\spei01_2010\spei01year_2009_01.nc'

    fileList = [x for x in os.listdir(dirPath) if x.endswith('.nc')]
    for idx, file in enumerate(fileList):
        fileFullPath = os.path.join(dirPath, file)

        outputFileName = file.split(".")[0] + ".tif"
        outputPath = os.path.join(outputDirPath, outputFileName)

        ReprojectImages2(fileFullPath,referencefilefilePath, outputPath)

main()