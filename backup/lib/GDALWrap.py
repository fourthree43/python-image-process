from osgeo import gdal, ogr, osr
import struct
import os


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
    dataSource = None
    band = None
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
    if os.path.exists(newRasterfn):
       os.remove(newRasterfn)
    outRaster = driver.Create(
        '' if newRasterfn is None else newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform(
        (originX, pixelWidth, 0, originY, 0, -pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.SetNoDataValue(-9999)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(EPSG)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    return outRaster
