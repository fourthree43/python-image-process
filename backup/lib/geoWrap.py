from osgeo import gdal
import os

from GDALWrap import readRaster, readNCRaster, array2raster2, array2raster

def wrapImg(baseCropFilePath, newRasterfn):
    # (1) 读取nc文件
    # baseCropFilePath = r"E:\testdata\riqimazie2010\Maize_DVS_C3S-glob-agric_2010_1_2010-01-10_dek_CSSF_hist_v1.nc"
    datasource = gdal.Open(baseCropFilePath)
    # (2) 读取元数据-原点坐标
    data = readRaster(baseCropFilePath)
    # newRasterfn = r"E:\testdata\riqimazie2010TIFF\Maize_DVS_C3S-glob-agric_2010_1_2010-01-10_dek_CSSF_hist_v1.tif"
    rasterOrigin = [-180, 90]
    pixelWidth = 0.1
    pixelHeight = 0.1
    EPSG = 4326
    array = data
    array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, EPSG, array)

def main():
    # 批处理
    dirPath = r"E:\climatecropdata\ERAcrop\dataset-sis-agroproductivity-indicators-maize2010"
    outputDirPath = r"E:\climatecropdata\ERAcrop\TIF\maize2010"

    fileList = [x for x in os.listdir(dirPath) if x.endswith('.nc')]
    for idx, file in enumerate(fileList):
        fileFullPath = os.path.join(dirPath, file)

        outputFileName = file.split(".")[0] + ".tif"
        outputPath = os.path.join(outputDirPath, outputFileName)

        wrapImg(fileFullPath, outputPath)

main()
