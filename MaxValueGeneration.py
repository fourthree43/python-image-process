'''
Author: 张仕山
Date: 2023-12-04 17:05:52
LastEditors: 张仕山
LastEditTime: 2023-12-10 22:12:25
Description:  最大值合成
FilePath: \MaxValueGeneration.py
'''
#!/usr/bin/env python
# --*-- encoding: utf-8 --*--

import os
import re
from os import path

from lib.GDALWrap import maxValueGeneration


"Maize_TWSO_C3S-glob-agric_2000_1_1999-04-20_dek_CSSF_hist_v1"


def MaxValueGeneration(inputPath):
  files = [x for x in os.listdir(inputPath) if x.endswith(".tif")]
  # 按年合成

  # 提取所有年份
  reExpression = r".+?(\d{4})_\d_\d{4}-\d{2}-\d{2}.+?"
  def getYear(text):
    m = re.match(reExpression, text)
    if m:
        return m.group(1)
    return ""
  
  years = sorted(list(set(list(map(getYear, files)))))

  def filterFileByYear(year):
    reExpression = '.+?%s_\\d_\\d{4}-\\d{2}-\\d{2}.+?' % year
    def tempFunc(test):
      if re.match(reExpression, test):
        return True
      else:
        return False
    return tempFunc
  
  for year in years:
    filteredFiles = list(filter(filterFileByYear(year), files))

    outputDirPath = path.join(inputPath, "result")
    if not path.exists(outputDirPath):
       os.mkdir(outputDirPath)

    fileName = os.path.splitext(path.basename(filteredFiles[0]))[0].split(year)[0]

    newRasterfn = os.path.join(os.path.abspath(
        outputDirPath), "{0}{1}_MaxValue.tif".format(fileName, year))

    maxValueGeneration(inputPath, filteredFiles, newRasterfn)
    print("{0}{1}最大值合成完成！".format(fileName, year))

if __name__ == "__main__":
  inputPath = r"F:\project\python\li\data\downloadall2\Maize\TWSO\result"
  MaxValueGeneration(inputPath)
