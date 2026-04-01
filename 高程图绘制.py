import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal 

# 中文显示设置（Mac）
plt.rcParams["font.family"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------- 读取 Excel ----------------------
def read_excel(file):
    sheet_data = {}
    xl = pd.ExcelFile(file)
    sheet_names = xl.sheet_names

    for idx, name in enumerate(sheet_names, 1):
        print(f"[{idx}/{len(sheet_names)}] 读取工作表：{name}")
        df = pd.read_excel(file, sheet_name=name, engine="openpyxl")
        sheet_data[name] = df.to_numpy()

    return sheet_names, sheet_data

# ---------------------- 绘制 高程剖面图 ----------------------

def read_dem(file_path):
    dataset = gdal.Open（file_path） # 打开文件
    band = dataset. GetRasterBand(1)
    ＃ 获取第一个波段
    dem_data = band. ReadAsArray 0
    ＃ 读取为NumPy数组
    geotransform = dataset. GetGeolransform（＃获取地理变换参数
    projection = dataset. CetProjection（ # 获取投影信息
    return dem_data, geotransform
def plot_elevation(sheet_names_list, sheet_data):
    plt.figure(figsize=(12, 5))

    for name in sheet_names_list:
        data = sheet_data[name]
        try:
            # 假设数据结构：第1列经度，第2列纬度，第3列高程
            lon = data[:, 0]
            lat = data[:, 1]
            ele = data[:, 2]

            # 计算距离（模拟路径长度）
            dist = np.arange(len(ele))

            # 绘制高程曲线
            plt.plot(dist, ele, linewidth=2, label=name)

        except Exception as e:
            print(f"⚠️ {name} 数据异常，跳过：{e}")

    plt.xlabel("路径距离（点序）", fontsize=12)
    plt.ylabel("高程（米）", fontsize=12)
    plt.title("秦直道及周边地形高程剖面图", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    file = 'result2.xlsx'
    sheet_names_list, sheet_data = read_excel(file)
    plot_elevation(sheet_names_list, sheet_data)