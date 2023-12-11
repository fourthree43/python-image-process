'''
Author: 张仕山
Date: 2023-12-05 09:40:34
LastEditors: 张仕山
LastEditTime: 2023-12-05 15:38:39
Description:  
FilePath: \trendAnalysis.py
'''
import os
import re
from lib.GDALWrap import readRaster, array2raster2, array2raster
import numpy as np

from functools import reduce

from tqdm import tqdm
import time

NoData = -10000

yearRe = r"(\d{4})"
# 按年份排序
def sortByTime(text):
   m = re.match(yearRe, text)
   if m:
      return m.group(1)
   return ""

def main(inputPath, outPath):
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
  
  n = len(fileList)
  # 年序列号
  yearSN = list(range(1, n + 1))

  print("计算趋势中...")
  # 趋势分析
  Sndvi = bandsData.sum(axis=0)
  Si = sum(yearSN)
  Si2 = sum([x * x for x in yearSN])

  npYearSN = np.asarray(yearSN, dtype='i1').reshape((n, 1, 1))
  newArr = np.tile(npYearSN, (1, *bandsData.shape[1:]))
  SiNDVIi = np.multiply(bandsData, newArr).sum(axis = 0)

  k = (n * SiNDVIi - Si * Sndvi) / (n * Si2 - Si * Si)

  # 输出
  print("结果输出中...")
  fileOutputPath = os.path.join(outPath, "TrendAnalysis_NDVI.tif")
  array2raster2(fileOutputPath, fileFullPath, k)
  print("计算完成！")

if __name__ == "__main__":
  inputPath = r"F:\project\python\li\data\modis"
  outPath = r"F:\project\python\li\result\modis"
  main(inputPath, outPath)
