import numpy as np
from datetime import datetime
import math
# import earthpy_plot_revised as ep

# def dataVisualization(raster_arr):
#     # 将图像的背景值设置为nan
#     raster_arr[raster_arr <= 0] = 'nan'
#     # 忽略nan值求最大和最小值 nanmin nanmax
#     min_DN = np.nanmin(raster_arr)
#     max_DN = np.nanmax(raster_arr)
#     plot_title = "数据可视化"
#     output_file_path_im = "./result/a.jpg"
#     # 栅格数据可视化
#     ep.plot_bands(raster_arr,
#                     title=plot_title,
#                     title_set=[25, 'bold'],
#                     cmap="seismic",
#                     cols=3,
#                     figsize=(12, 12),
#                     extent=None,
#                     cbar=True,
#                     scale=False,
#                     vmin=-16,
#                     vmax=45,
#                     ax=None,
#                     alpha=1,
#                     norm=None,
#                     save_or_not=True,
#                     save_path=output_file_path_im,
#                     dpi_out=600,
#                     bbox_inches_out="tight",
#                     pad_inches_out=0.1,
#                     text_or_not=True,
#                     text_set=[0.75, 0.95, "T(°C)", 20, 'bold'],
#                     colorbar_label_set=True,
#                     label_size=20,
#                     cbar_len=2,
#                     )  

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