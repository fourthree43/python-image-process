'''
Author: 张仕山
Date: 2023-09-11 17:48:35
LastEditors: 张仕山
LastEditTime: 2023-12-11 18:56:53
Description:  
FilePath: \lib\GDALWrap.py
'''
import sys
import math
import functools
from osgeo import gdal, ogr, osr
import netCDF4 as nc
import struct
import os
from os import path

import re
from tqdm import tqdm
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from pandas import ExcelWriter


os.environ['PROJ_LIB'] = r'D:\ProgramFiles\anaconda3\pkgs\proj-6.2.1-h3758d61_0\Library\share\proj'
noData = -9999

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

def array2rasterByRefImg(newRasterfn, refRasterfn, array, noData=0):
    # 获取源影像的变换和空间坐标系参考
    source = gdal.Open(refRasterfn)
    geoTransform = source.GetGeoTransform()
    spatialRef = source.GetSpatialRef()

    shape = array.shape
    if len(shape) == 2:
      rows = array.shape[0]
      cols = array.shape[1]
      bands = 1
    elif len(shape) == 3:
      bands = min(*shape)
      idx = list(shape).index(bands)
      if idx == 0:
        rows = array.shape[1]
        cols = array.shape[2]
      elif idx == 2:
        rows = array.shape[0]
        cols = array.shape[1]

    # 创建输出影像
    if newRasterfn is None:
      driver = gdal.GetDriverByName("MEM")
    else:
      driver = gdal.GetDriverByName('GTiff')

    if os.path.exists(newRasterfn):
        os.remove(newRasterfn)

    if array.dtype.kind == "f":
      dataType = gdal.GDT_Float32

    outRaster = driver.Create(
        '' if newRasterfn is None else newRasterfn, cols, rows, bands, dataType)
    # 设置输出影像的变换和空间坐标系参考
    outRaster.SetGeoTransform(geoTransform)
    outRaster.SetProjection(spatialRef.ExportToWkt())

    # 写入影像数据
    for i in range(bands):
      outband = outRaster.GetRasterBand(i+1)
      outband.SetNoDataValue(noData)
      if len(shape) == 3:
        if idx == 0:
            outband.WriteArray(array[i, :, :])
        elif idx == 2:
          outband.WriteArray(array[:, :, i])
      else:
        outband.WriteArray(array)
      outband.FlushCache()
    source = None
    outRaster = None

def array2raster2(newRasterfn, mirrorImg, array, noValue = noData):
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
    outband.SetNoDataValue(noValue)
    outband.WriteArray(array)

    outband.FlushCache()
    source = None

def array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, EPSG, array):
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]

    if newRasterfn is None:
      driver = gdal.GetDriverByName("MEM")
    else:
      driver = gdal.GetDriverByName('GTiff')
    if os.path.exists(newRasterfn):
       os.remove(newRasterfn)

    shape = array.shape
    sz = len(shape)
    bands = 1
    rows = shape[0]
    cols = shape[1]
    if sz == 3:
      bands, rows, cols = shape

    outRaster = driver.Create(
        '' if newRasterfn is None else newRasterfn, cols, rows, bands, gdal.GDT_Float32)

    outRaster.SetGeoTransform(
        (originX, pixelWidth, 0, originY, 0, -pixelHeight))

    sz = len(array.shape)
    if sz == 3:
      for i in range(bands):
        outband = outRaster.GetRasterBand(i + 1)
        outband.SetNoDataValue(noData)
        outband.WriteArray(array[i, :, :].reshape((rows, cols)))
    else:
      outband = outRaster.GetRasterBand(1)
      outband.SetNoDataValue(noData)
      outband.WriteArray(array)

    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(EPSG)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    return outRaster


yearRe = r"(\d{4})"
# 按年份排序 
def sortByTime(text, reExpression = yearRe):
   m = re.match(reExpression, text)
   if m:
      return m.group(1)
   return ""

def readImgDataByDirPath(inputPath, extension, sortByTime=sortByTime):
    fileList = [x for x in os.listdir(inputPath) if x.endswith(extension)]
    # 文件排序和分组
    fileList = sorted(fileList, key=sortByTime, reverse=False)  # 年份升序排列
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
    return bandsData


def readImgDataByFileList(inputPath, fileList):
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
    return bandsData


def calculateValueByAxis(bandsData, calcFuncHandler, msg="开始计算", axis=0):
  shape = bandsData.shape
  rows = shape[1]
  colmns = shape[2]
  bands = shape[0]
  result = np.full((rows, colmns), noData, dtype="f")
  barMsg = tqdm(range(rows))
  barMsg.set_description(msg)

  for row in barMsg:
      for col in range(colmns):
          x = []
          if axis == 0:
            x = bandsData[:, row, col].reshape(bands)
          elif axis == 1:
            x = bandsData[row, :, col].reshape(bands)
          elif axis == 2:
            x = bandsData[row, col, :].reshape(bands)
          nanLen = x[np.isnan(x)].shape[0]
          noDataLen = x[x == noData].shape[0]

          # 如果序列存在noData的话
          # if nanLen != 0 or noDataLen != 0:
          #   pass
          # else:
          if not(nanLen == x.size or noDataLen == x.size):
          # if nanLen != 0 or noDataLen != 0:
            try:
              hindex = calcFuncHandler(x)
              result[row, col] = hindex
            except Exception as e:
              print(x)
  
  return result


def calculateValueByPixel(bandsData, calcFuncHandler, msg="开始计算"):
  shape = bandsData.shape
  rows = shape[0]
  colmns = shape[1]
  result = np.full((rows, colmns), noData, dtype="f")
  barMsg = tqdm(range(rows))
  barMsg.set_description(msg)

  for row in barMsg:
      for col in range(colmns):
        try:
          hindex = calcFuncHandler(bandsData[row, col])
          result[row, col] = hindex
        except Exception as e:
          print(bandsData[row, col])
  return result

# 分块处理
def BlockingComputing(num):
  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
      v_arr = args[0]
      (bands, rows, cols) = v_arr.shape
      result = np.zeros((rows, cols), dtype="f")
      iStep = math.ceil(rows / num)
      jStep = math.ceil(cols / num)
      for i in tqdm(range(0, num), desc="执行分块计算"):
        for j in tqdm(range(0, num), desc="分块计算...", leave=False):
          subResult = func(v_arr[:, i * iStep: (i + 1) *
                                 iStep, j * jStep: (j + 1) * jStep])
          result[i * iStep: (i + 1) *
                iStep, j * jStep: (j + 1) * jStep] = subResult
      return result
    return wrapper
  return decorator


def maxValueGeneration(inputPath, imgList, outputImgPath):
   datas = readImgDataByFileList(inputPath, imgList)
   result = np.max(datas, axis=0)
   array2rasterByRefImg(outputImgPath, path.join(
       inputPath, imgList[0]), result, noData)


def reprojectLayer(input):
  driver = ogr.GetDriverByName('ESRI Shapefile')
  inDataSet = driver.Open(input)

  # input SpatialReference
  inLayer = inDataSet.GetLayer()
  inSpatialRef = inLayer.GetSpatialRef()

  # output SpatialReference
  outSpatialRef = osr.SpatialReference()
  outSpatialRef.ImportFromEPSG(4326)

  # create the CoordinateTransformation
  coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

  # create the output layer
  outputShapefile = r'./result/temp.shp'
  if os.path.exists(outputShapefile):
      driver.DeleteDataSource(outputShapefile)

  outDataSet = ogr.GetDriverByName(
      'ESRI Shapefile').CreateDataSource(outputShapefile)
  outLayer = outDataSet.CreateLayer("basemap_4326", geom_type=ogr.wkbPolygon)

  # add fields
  inLayerDefn = inLayer.GetLayerDefn()
  for i in range(0, inLayerDefn.GetFieldCount()):
      fieldDefn = inLayerDefn.GetFieldDefn(i)
      outLayer.CreateField(fieldDefn)

  # get the output layer's feature definition
  outLayerDefn = outLayer.GetLayerDefn()

  # loop through the input features
  inFeature = inLayer.GetNextFeature()
  while inFeature:
      # get the input geometry
      geom = inFeature.GetGeometryRef()
      # reproject the geometry
      geom.Transform(coordTrans)
      # create a new feature
      outFeature = ogr.Feature(outLayerDefn)
      # set the geometry and attribute
      outFeature.SetGeometry(geom)

      # for i in range(0, outLayerDefn.GetFieldCount()):
      #     outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))

      # add the feature to the shapefile
      outLayer.CreateFeature(outFeature)
      # dereference the features and get the next input feature
      outFeature = None
      inFeature = inLayer.GetNextFeature()

  # Save and close the shapefiles
  inDataSet = None
  outLayer = None
  # outDataSet = None
  return outDataSet


def vector2list(input):
  shpDriver = ogr.GetDriverByName("ESRI Shapefile")
  dataSource = shpDriver.Open(input)
  layer = dataSource.GetLayer(0)
  result = []
  for feature in layer:
    geometry = feature.GetGeometryRef()
    geomType = geometry.GetGeometryType()
    if geomType == ogr.wkbPoint:
      result.append(list(geometry.GetPoints()[0]))
    elif geomType == ogr.wkbPolygon:
      result.append(list(geometry.GetPoints()[0]))
  layer.ResetReading()

  dataSource = None
  layer = None
  return result


def getExtentWithLatLon(input):
  shpDriver = ogr.GetDriverByName("ESRI Shapefile")
  dataSource = shpDriver.Open(input)
  layer = dataSource.GetLayer(0)
  result = []
  for feature in layer:
    geometry = feature.GetGeometryRef()
    # 转成wgs84坐标
    # 坐标转换
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    source = layer.GetSpatialRef()
    transform = osr.CoordinateTransformation(source, target)
    geometry.Transform(transform)
    # minx: float
    # maxx: float
    # miny: float
    # maxy: float
    minx, maxx, miny, maxy = geometry.GetEnvelope()
    return minx, maxx, miny, maxy

  layer.ResetReading()

  dataSource = None
  layer = None
  return [0, 0, 0, 0]

# 写矢量图层
def writeVectorLayerByGeometry(outputPath, geometry, type=ogr.wkbPolygon):
    if type is None:
       type = geometry.GetGeometryType()
    # if srs is None:
    #    srs = geometry.GetSpatialReference()
    srs = geometry.GetSpatialReference()

    # Create the output shapefile
    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outputPath):
        shpDriver.DeleteDataSource(outputPath)

    # create the layer
    outDataSource = shpDriver.CreateDataSource(outputPath)
    outLayer = outDataSource.CreateLayer(outputPath, srs, geom_type=type)

    # Create the feature and set values
    featureDefn = outLayer.GetLayerDefn()
    outFeature = ogr.Feature(featureDefn)
    outFeature.SetGeometry(geometry)

    outLayer.CreateFeature(outFeature)
    outFeature = None
    outDataSource = None

# 写矢量图层
def writeVectorLayerByList(outputPath, EPSG, positions, type=ogr.wkbPoint, fields=[{}]):
    # Create the output shapefile
    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outputPath):
        shpDriver.DeleteDataSource(outputPath)

    # create the spatial reference
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)
    # create the layer
    outDataSource = shpDriver.CreateDataSource(outputPath)
    outLayer = outDataSource.CreateLayer(outputPath, srs, geom_type=type)

    # create a field
    filedIsNotEmpty = len(fields[0].items()) > 0
    if filedIsNotEmpty:
        for (key, value) in fields[0].items():
            if isinstance(value, str):
                idField = ogr.FieldDefn(key, ogr.OFTString)
                outLayer.CreateField(idField)
            elif isinstance(value, int):
                idField = ogr.FieldDefn(key, ogr.OFTInteger64)
                outLayer.CreateField(idField)

    # Create the feature and set values
    featureDefn = outLayer.GetLayerDefn()
    for i, pos in enumerate(positions):
        outFeature = ogr.Feature(featureDefn)
        if type == ogr.wkbPoint:
          # create geometry
          point = ogr.Geometry(type)
          point.AddPoint(pos[0], pos[1])
          outFeature.SetGeometry(point)
        elif type == ogr.wkbLineString:
          # Create ring
          # Create ring
          line = ogr.Geometry(ogr.wkbLinearRing)
          for j in pos:
            line.AddPoint(*j)
          outFeature.SetGeometry(line)

          # Create polygon
          poly = ogr.Geometry(ogr.wkbPolygon)
          poly.AddGeometry(ring)
        elif type == ogr.wkbPolygon:
          # Create ring
          ring = ogr.Geometry(ogr.wkbLinearRing)
          for j in pos:
            ring.AddPoint(*j)
          # Create polygon
          poly = ogr.Geometry(ogr.wkbPolygon)
          poly.AddGeometry(ring)
          outFeature.SetGeometry(poly)

        if filedIsNotEmpty:
            for (fieldName, fieldValue) in fields[i].items():
                outFeature.SetField(fieldName, fieldValue)

        outLayer.CreateFeature(outFeature)
        outFeature = None
    outDataSource = None

def getMaskArray(regionFilePath, regionGeometry, outputTifffile, imgFilePath):
  # Define pixel_size and NoData value of new raster
  NoData_value = 0

  # Open the data source and read in the extent
  # source_ds = ogr.Open(regionFilePath)
  # source_layer = source_ds.GetLayer()
  # source_srs = source_layer.GetSpatialRef()
  # x_min, x_max, y_min, y_max = source_layer.GetExtent()
  x_min, x_max, y_min, y_max = regionGeometry.GetEnvelope()
  outDataSet = ogr.GetDriverByName('ESRI Shapefile').Open(regionFilePath)
  inLayer = outDataSet.GetLayer()

  # 按照imgFilePath输出
  dataSource = gdal.Open(imgFilePath)
  geoTransform = dataSource.GetGeoTransform()
  pixel_size = geoTransform[1]
  xPoi = geoTransform[0]
  yPoi = geoTransform[3]

  # Create the destination data source
  y_res = int((y_max - y_min) / pixel_size)
  x_res = int((x_max - x_min) / pixel_size)

  tifDriver = gdal.GetDriverByName('GTiff')
  if os.path.exists(outputTifffile):
      # tifDriver.Delete(outputTifffile)
      os.remove(outputTifffile)
  target_ds = tifDriver.Create(outputTifffile, x_res, y_res, gdal.GDT_Byte)

  # 计算对应的左上角位置
  realX = (int(x_min / pixel_size) + 1) * pixel_size
  realY = (int(y_max / pixel_size) + 1) * pixel_size
  target_ds.SetGeoTransform((realX, pixel_size, 0, realY, 0, -pixel_size))
  band = target_ds.GetRasterBand(1)
  band.SetNoDataValue(NoData_value)

  targetSrs = dataSource.GetSpatialRef()
  target_ds.SetSpatialRef(targetSrs)

  # Rasterize
  gdal.RasterizeLayer(target_ds, [1], inLayer, burn_values=[1])

  # Read as array
  array = band.ReadAsArray()

  xStart = int((x_min - xPoi) / pixel_size)
  yStart = int((yPoi - y_max) / pixel_size)
  target_ds = None
  band = None
  return array, [xStart, yStart, x_res, y_res]

def getMaskArrayBackup(regionFilePath, imgFilePath):
  # Define pixel_size and NoData value of new raster
  NoData_value = 0

  # Open the data source and read in the extent
  # source_ds = ogr.Open(regionFilePath)
  # source_layer = source_ds.GetLayer()
  # source_srs = source_layer.GetSpatialRef()
  # x_min, x_max, y_min, y_max = source_layer.GetExtent()
  x_min, x_max, y_min, y_max = getExtentWithLatLon(regionFilePath)

  outDataSet = reprojectLayer(regionFilePath)
  inLayer = outDataSet.GetLayer()

  # 按照imgFilePath输出
  dataSource = gdal.Open(imgFilePath)
  geoTransform = dataSource.GetGeoTransform()
  pixel_size = geoTransform[1]

  # Create the destination data source
  y_res = int((y_max - y_min) / pixel_size)
  x_res = int((x_max - x_min) / pixel_size)

  outputTifffile = r'./result/mask.tif'
  tifDriver = gdal.GetDriverByName('GTiff')
  if os.path.exists(outputTifffile):
      # tifDriver.Delete(outputTifffile)
      os.remove(outputTifffile)
  target_ds = tifDriver.Create(outputTifffile, x_res, y_res, gdal.GDT_Byte)

  # 计算对应的左上角位置
  realX = (int(x_min / 0.25) + 1) * 0.25
  realY = (int(y_max / 0.25) + 1) * 0.25
  target_ds.SetGeoTransform((realX, pixel_size, 0, realY, 0, -pixel_size))
  band = target_ds.GetRasterBand(1)
  band.SetNoDataValue(NoData_value)

  targetSrs = osr.SpatialReference()
  targetSrs.ImportFromEPSG(4326)
  target_ds.SetSpatialRef(targetSrs)

  # Rasterize
  gdal.RasterizeLayer(target_ds, [1], inLayer, burn_values=[1])

  # Read as array
  array = band.ReadAsArray()
  return array, realX, realY

def zonalStats(input_zone_polygon, input_value_raster, output_img):
    # Open data
    raster = gdal.Open(input_value_raster)

    shp = ogr.Open(input_zone_polygon)
    lyr = shp.GetLayer()

    # Get raster georeference info
    transform = raster.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]

    # Reproject vector geometry to same projection as raster
    sourceSR = lyr.GetSpatialRef()
    targetSR = osr.SpatialReference()
    targetSR.ImportFromWkt(raster.GetProjectionRef())
    coordTrans = osr.CoordinateTransformation(sourceSR,targetSR)

    # # feat = lyr.GetNextFeature()
    # geom = feat.GetGeometryRef()
    # geom.Transform(coordTrans)

    # # Get extent of feat
    # # geom = feat.GetGeometryRef()

    feat = lyr.GetNextFeature()
    geom = feat.GetGeometryRef()
    geom.Transform(coordTrans)
    if (geom.GetGeometryName() == 'MULTIPOLYGON'):
        count = 0
        pointsX = []; pointsY = []
        for polygon in geom:
            geomInner = geom.GetGeometryRef(count)
            ring = geomInner.GetGeometryRef(0)
            numpoints = ring.GetPointCount()
            for p in range(numpoints):
                    lat, lon, z = ring.GetPoint(p)
                    pointsX.append(lon)
                    pointsY.append(lat)
            count += 1
    elif (geom.GetGeometryName() == 'POLYGON'):
        ring = geom.GetGeometryRef(0)
        numpoints = ring.GetPointCount()
        pointsX = []; pointsY = []
        for p in range(numpoints):
                lat, lon, z = ring.GetPoint(p)
                pointsX.append(lon)
                pointsY.append(lat)

    else:
        sys.exit("ERROR: Geometry needs to be either Polygon or Multipolygon")

    xmin = min(pointsX)
    xmax = max(pointsX)
    ymin = min(pointsY)
    ymax = max(pointsY)

    # Specify offset and rows and columns to read
    xoff = int((xmin - xOrigin)/pixelWidth)
    yoff = int((yOrigin - ymax)/pixelWidth)
    xcount = int((xmax - xmin)/pixelWidth)+1
    xcount = raster.RasterXSize if xcount > raster.RasterXSize else xcount
    ycount = int((ymax - ymin)/pixelWidth)+1

    # Create memory target raster
    target_ds = None
    if len(output_img) > 0:
      target_ds = gdal.GetDriverByName('GTiff').Create(
          output_img, xcount, ycount, 1, gdal.GDT_Byte)
    else:
      target_ds = gdal.GetDriverByName('MEM').Create(
          '', xcount, ycount, 1, gdal.GDT_Byte)

    target_ds.SetGeoTransform((
        xmin, pixelWidth, 0,
        ymax, 0, pixelHeight,
    ))

    # Create for target raster the same projection as for the value raster
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster.GetProjectionRef())
    target_ds.SetProjection(raster_srs.ExportToWkt())

    # Rasterize zone polygon to raster
    gdal.RasterizeLayer(target_ds, [1], lyr, burn_values=[1])

    # Read raster as arrays
    banddataraster = raster.GetRasterBand(1)
    dataraster = banddataraster.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float64)

    bandmask = target_ds.GetRasterBand(1)
    datamask = bandmask.ReadAsArray(0, 0, xcount, ycount).astype(np.float64)

    # Mask zone of raster
    zoneraster = np.ma.masked_array(dataraster,  np.logical_or(
        np.logical_not(datamask), dataraster == noData))
    # zoneraster = np.ma.masked_array(dataraster,  np.logical_not(datamask))

    # Calculate statistics of zonal raster
    return np.mean(zoneraster),np.max(zoneraster), np.min(zoneraster),np.std(zoneraster),np.var(zoneraster) # np.median(zoneraster)

def loopZonalStats(input_zone_polygon, input_value_raster, outputFilePath, fieldName = "FCNAME"):
    shp = ogr.Open(input_zone_polygon)
    lyr = shp.GetLayer()
    featList = range(lyr.GetFeatureCount())
    statDict = {}

    dirPath = path.dirname(input_zone_polygon)
    outputDirPath = path.join(dirPath, "result")

    if not path.exists(outputDirPath):
       os.mkdir(outputDirPath)

    fileName = os.path.splitext(path.basename(input_zone_polygon))[0]

    df = DataFrame([], columns=['ID', '国家', '均值', '最大值', '最小值', '标准差', '方差'])
    barMsg = tqdm(featList)
    barMsg.set_description("统计中...")
    idx = 0
    for FID in barMsg:
        feat = lyr.GetFeature(FID)
        fieldValue = feat.GetField(fieldName)
        if fieldValue is None:
           continue
        barMsg.set_description("统计中({0}))".format(fieldValue))

        outputFile = os.path.join(os.path.abspath(
            outputDirPath), "{0}-{1}.shp".format(fileName, fieldValue))
        # idx = 1
        # while path.exists(outputFile):
        #    outputFile = os.path.join(os.path.abspath(
        #        outputDirPath), "{0}-{1}-{2}.shp".format(fileName, fieldValue, idx))
        #    idx += 1
        if not path.exists(outputFile):
           writeVectorLayerByGeometry(outputFile, feat.GetGeometryRef())

        outputFileRaster = os.path.join(os.path.abspath(
            outputDirPath), "{0}-{1}-mask.tif".format(fileName, fieldValue))

        meanValue = zonalStats(outputFile,
                               input_value_raster, outputFileRaster)
        statDict[FID] = meanValue
        
        data = {
          'ID': idx + 1,
          '国家': fieldValue,
          '均值': meanValue[0],
          '最大值': meanValue[1],
          '最小值': meanValue[2],
          '标准差': meanValue[3],
          '方差': meanValue[4]
        }
        df.loc[idx] = [idx + 1, fieldValue, *meanValue]

        idx += 1
    with ExcelWriter(outputFilePath,mode='w') as writer:
      df.to_excel(writer,sheet_name='统计表')
    return statDict

# 批量处理
def BatchingProcessor(inputPath, extension):
  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
      files = [x for x in os.listdir(inputPath) if x.endswith(extension)]

      msgBar = tqdm(files, desc="批量计算...", leave=False)
      for j in msgBar:
        imgPath = path.join(path.abspath(inputPath), j)
        fileName = path.splitext(j)[0]
        func(imgPath)
        msgBar.set_description(f'文件数据处理中（{fileName}）')
      # msgBar.close()
      return None
    return wrapper
  return decorator
