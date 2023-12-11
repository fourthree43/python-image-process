'''
Author: 张仕山
Date: 2023-12-09 16:05:08
LastEditors: 张仕山
LastEditTime: 2023-12-11 18:38:05
Description:  
'''
import os
from os import path

from lib.GDALWrap import loopZonalStats


def ZonelStatistics(inputPathRaster, inputPathVector, outPath):
  # 文件排序和分组
  loopZonalStats(inputPathVector, inputPathRaster, outPath)

if __name__ == "__main__":
  inputPathRaster = r"F:\project\python\li\data\statistics\Maize_TWSO_C3S-glob-agric_2020_MaxValue.tif"
  inputPathVector = r"F:\project\python\li\data\statistics\WORD\World_countries.shp"

  dirPath = path.dirname(inputPathRaster)
  outputDirPath = path.join(dirPath, "result")
  if not path.exists(outputDirPath):
      os.mkdir(outputDirPath)
  outPath = path.join(outputDirPath, "统计表.xlsx")

  ZonelStatistics(inputPathRaster, inputPathVector, outPath)
