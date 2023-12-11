'''
Author: 张仕山
Date: 2023-12-09 16:05:08
LastEditors: 张仕山
LastEditTime: 2023-12-10 22:44:46
Description:  
FilePath: \ExtractPhenology.py
'''
import os
import re
from os import path
import numpy as np
import datetime as dt

from lib.GDALWrap import readImgDataByFileList, array2rasterByRefImg, calculateValueByAxis, noData, calculateValueByPixel

"Maize_DVS_C3S-glob-agric_2000_1_1999-04-20_dek_CSSF_hist_v1"
def filterFileByYear(year):
    reExpression = '.+?%s_\\d_\\d{4}-\\d{2}-\\d{2}.+?' % year
    def tempFunc(test):
      if re.match(reExpression, test):
        return True
      else:
        return False
    return tempFunc


yearRe = r".+?(\d{4}-\d{2}-\d{2}).+?"
# yearRe = r"(\d{4}-\d{2}-\d{2})"
# 按年份排序
def sortByTime(text, reExpression=yearRe):
   m = re.match(reExpression, text)
   if m:
      return m.group(1)
   return ""

def ExtractPhenology(inputPath, outPath, val, baseYear):
  files = [x for x in os.listdir(inputPath) if x.endswith(".tif")]
  # 过滤年份
  filteredFiles = list(filter(filterFileByYear(baseYear), files))
  filteredFiles = sorted(filteredFiles, key=sortByTime,
                         reverse=False)  # 年份升序排列
  # 提取固定年份的物候
  datas = readImgDataByFileList(inputPath, filteredFiles)
  # 找到对应物候值对应的文件索引
  def extractFileIndex(x):
     if val == 0:
      idxs = np.where(x > val)[0]
      len = idxs.shape[0]
      if len == 0:
          return -1
      return idxs[0]
     else:
      idxs = np.where(x >= val)[0]
      len = idxs.shape[0]
      if len == 0:
          return -1
      return idxs[0]
  fileIndex = calculateValueByAxis(datas, extractFileIndex,
                       msg="开始计算对应物候值对应的文件索引...", axis=0)

  # 索引对应时间时间
  reVal = re.compile(yearRe)
  def dayMapFunc(x):
    if x == -1 or x <= 0 or x == noData:
       return noData
    fileName = filteredFiles[int(x)]
    m = reVal.match(fileName)
    if m:
      timeStr = m.group(1)
      if int(timeStr[0: 4]) == baseYear:
        base_time = dt.datetime.strptime(timeStr, '%Y-%m-%d')
        return base_time.timetuple().tm_yday
      else:
        base_time = dt.datetime.strptime(timeStr, '%Y-%m-%d')
        return base_time.timetuple().tm_yday - 365
    else:
      return noData

  result = calculateValueByPixel(fileIndex, dayMapFunc,
                                   msg="开始计算物候期...")
  # 输出
  array2rasterByRefImg(outPath, path.join(
      inputPath, filteredFiles[0]), result, noData)
  print("计算完成！")

if __name__ == "__main__":
  inputPath = r"F:\project\python\li\data\downloadall2\Maize\DVS\result"

  outputDirPath = path.join(inputPath, "result")
  if not path.exists(outputDirPath):
      os.mkdir(outputDirPath)

  base = 2020
  val = 1.5 # 0 - 2某个物候期
  outPath = path.join(outputDirPath, "Maize_DVS_C3S-glob-agric_phenology_{0}_{1}.tif".format(base, val))

  ExtractPhenology(inputPath, outPath, val, base)
