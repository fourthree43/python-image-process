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
import os
import math
import re
from datetime import datetime

from lib.GDALWrap import readRaster, readNCRaster, array2raster2, array2raster
from lib.common import day2Month, calculateDroughtIndex

os.environ['PROJ_LIB'] = r'D:\Program Files (x86)\anaconda\pkgs\proj-6.2.1-h3758d61_0\Library\share\proj'

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
def calculateTimeLine(array):
    begin = None
    end = None
    idx = 0

    # False: 正常情况, True: 引入第二份物候数据
    flag = False

    # try:
    #     a = array.tolist()
    #     if abs(a[9] - 0.177145) < 0.0001 and abs(a[11] - 0.488213) < 0.0001 and abs(a[14] - 0.959656) < 0.0001:
    #         print(11, array)
    # except ValueError as e:
    #     pass

    for x in np.nditer(array):

        # 判断是否存在一开始就已经成熟的种植区域
        if abs(x - 2.0) < 0.000001 and x is None:
            print(x, idx)
            print(array)
        # # 如果刚开始就大于0.3
        # if (array[0] >= 0.3) and x is None:
        #     print(11111, array)


        # if x > 0.5 and begin is None and x < 2.0:
        #     # 检索作物开始种植的日期
        #     begin = idx

        if x > 0 and begin is None and x < 2.0:
            if array[0] > 0:
                flag = True
            # 检索作物开始种植的日期
            begin = idx
            idx += 1
            continue
        # 当年达到成熟期2.0
        if end is None and not (begin is None) and x == 2.0:
            end = idx
            return begin, end, flag
        # 到达终点
        if end is None and not (begin is None) and idx == array.shape[0] - 1:
            end = idx
            return begin, end, flag
        # 当年未达到成熟期2.0
        if end is None and not (begin is None) and abs(x - array[idx + 1]) < 0.000001:
            end = idx
            return begin, end, flag
        idx += 1
    return begin, end, flag

def main():
    # (1) 读取物候和SPEI的数据
    baseCropFilePath = r"E:\climatecropdata\ERAcrop\TIF\maize2010"
    baseSPEIFilePath = r"D:\data\spei\spei06\spei06_2010mon"

    baseYear = 2010
    txtFormat = "spei06year_{0}_{1}.nc"
    datasets = {}

    outPath = r"D:\data\spei\spei01Eracol"
    noData = -9999
    #（1）读取物候信息
    fileList = [x for x in os.listdir(baseCropFilePath) if x.endswith('.tif')]
    bandsData = None
        
    for idx, file in enumerate(fileList):
        fileFullPath = os.path.join(baseCropFilePath, file)
        band = readNCRaster(fileFullPath)
        if bandsData is None:
            bandsData = np.zeros((*band.shape, len(fileList)), dtype=float)
        bandsData[:,:,idx] = band

    rows, columns, d = bandsData.shape
    # 输出文件
    resultCropD = np.zeros(band.shape, dtype = float, order = 'C')
    resultCropD[::] = noData
    resultCropS = np.zeros(band.shape, dtype = float, order = 'C')
    resultCropD[::] = noData
    resultCropF = np.zeros(band.shape, dtype = float, order = 'C')
    resultCropD[::] = noData
    # 定义提取日期的正则表达式
    reRepr = r"\d{4}-\d{2}-\d{2}"


    # 新物候数据
    prevYearWuhou = r"E:\climatecropdata\allcrop2010\cropcal\maize1plantday.tif"
    prevYearWuhouData = readRaster(prevYearWuhou)
    #prevYearWuhou2 = r"C:\Users\yy\Desktop\python\version2.0\data\maize2plantday.tif"
   # prevYearWuhouData2 = readRaster(prevYearWuhou2)

    for i in range(rows):
        for j in range(columns):
            timeLine = bandsData[i, j, :]
            idx = np.where(timeLine >= 0)
            if (idx[0].shape[0] > 0):
                # print(timeLine)
                # 获取当前像元物候信息 (种植月份和成熟月份)
                begin, end, flag = calculateTimeLine(timeLine) # 开始和结束文件的索引
                if begin is None or end is None:
                    # print(i, j, timeLine.tolist())
                    continue
                # 文件索引转月份
                matchObj = re.search(reRepr, fileList[begin]) 
                if matchObj:
                    dateStr = matchObj.group()
                    dateArr = dateStr.split("-")
                    beginMonth = int(dateArr[1])
                    # if int(dateArr[2]) <= 15:
                    #     beginMonth = int(dateArr[1])
                    # else:
                    #     beginMonth = int(dateArr[1]) + 1

                matchObj = re.search(reRepr, fileList[end]) 
                if matchObj:
                    dateStr = matchObj.group()
                    dateArr = dateStr.split("-")
                    # if int(dateArr[2]) <= 15:
                    #     endMonth = int(dateArr[1]) - 1
                    # else:
                    #     endMonth = int(dateArr[1])
                    endMonth = int(dateArr[1])
                # 遍历读取对应物候内的SPEI
                bands = []
                for month in range(beginMonth, endMonth + 1):
                    fileName = txtFormat.format(baseYear, str(month).rjust(2, '0'))
                    fileFullPath = os.path.join(baseSPEIFilePath, fileName)
                    if os.path.exists(fileFullPath):
                        if datasets.get(fileFullPath) is None:
                            banda = readNCRaster(fileFullPath)
                            datasets[fileFullPath] = banda
                        else:
                            banda = datasets.get(fileFullPath)
                        bands.append(banda)
                    else: 
                        print("文件不存在：{0}".format(fileFullPath))
                
                # 获取对应位置索引
                rate = band.shape[0] / banda.shape[0]
                x = math.ceil((i + 1) / rate) - 1
                y = math.ceil((j + 1) / rate) - 1

                datas = list(map(lambda arr: arr[x, y], bands))
                datas1 = []
                if flag:
                    # 引入第二份物候数据，重新计算开始种植年份
                    # rate1 =  band.shape[0] / prevYearWuhouData.shape[0]
                    # x1 = math.ceil((i + 1) / rate1) - 1
                    # y1 = math.ceil((j + 1) / rate1) - 1
                    # days = prevYearWuhouData[x1, y1]
                    # beginMonth1 = day2Month(days, baseYear - 1, False)
                    # bands1 = []
                    # if beginMonth1 < 6:
                    #     days = prevYearWuhouData2[x1, y1]
                    #     if days > 180 and days < 366:
                    #         beginMonth1 = day2Month(days, baseYear - 1, False)
                    #     else:
                    #         beginMonth1 = 13 - (endMonth - beginMonth + 1) / (2 - timeLine[0]) * timeLine[0]
                    #         if beginMonth1 < 6 and beginMonth1 > 12:
                    #             beginMonth1 = 10

                    # beginMonth1 = 13 - (endMonth - beginMonth + 1) / (2 - timeLine[0]) * timeLine[0]
                    bands1 = []
                    d = 2 / 6
                    num = round(timeLine[0] / d)
                    beginMonth1 = 13 - num
                    print(111, beginMonth1, timeLine[0])

                    for month in range(beginMonth1, 12 + 1):
                        fileName = txtFormat.format(baseYear - 1, str(month).rjust(2, '0'))
                        fileFullPath = os.path.join(baseSPEIFilePath, fileName)
                        if os.path.exists(fileFullPath):
                            if datasets.get(fileFullPath) is None:
                                banda = readNCRaster(fileFullPath)
                                datasets[fileFullPath] = banda
                            else:
                                banda = datasets.get(fileFullPath)
                            bands1.append(banda)
                        else: 
                            print("文件不存在：{0}".format(fileFullPath))
                    
                    datas1 = list(map(lambda arr: arr[x, y], bands1))
                datas = datas1 + datas
                # 计算当前位置指标
                D, S, F = calculateDroughtIndex(datas)
                # print(i, j, D, S, F)
                resultCropD[i, j] = D
                resultCropS[i, j] = S
                resultCropF[i, j] = F    

    # # (1) 定位第一个文件
    # prevYearWuhou = r"C:\Users\yy\Desktop\python\version2.0\data\maize1plantday.tif"
    # prevYearWuhouData = readRaster(prevYearWuhou)

    # firstCrop = bandsData[:,:,0]
    # cropIdx = np.where(firstCrop > 0)
    # [xCrop, yCrop] = cropIdx
    # for idx, x in enumerate(xCrop.tolist()):
    #     # firstCrop[x, yCrop[idx]]
    #     rate = prevYearWuhouData.shape[0] / firstCrop.shape[0]
    #     x1 = math.ceil((x + 1) / rate) - 1
    #     y1 = math.ceil((yCrop[idx] + 1) / rate) - 1
    #     days = prevYearWuhouData[x1, y1]
    #     beginMonth = day2Month(days, baseYear, False)

        


    array2raster(os.path.join(os.path.abspath(outPath), 'SPEI06maize_Intensity_2010.tif'), [-180, 90], 0.1, 0.1, 4326, resultCropS)
    array2raster(os.path.join(os.path.abspath(outPath), 'SPEI06maize_Frequency_2010.tif'), [-180, 90], 0.1, 0.1, 4326, resultCropF)     
    array2raster(os.path.join(os.path.abspath(outPath), 'SPEI06maize_Duration_2010.tif'), [-180, 90], 0.1, 0.1, 4326, resultCropD)     

if __name__ == '__main__':
    print("开始执行...")
    main()
    print("执行完成...")
