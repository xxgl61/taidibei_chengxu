import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ================= 中文显示（Mac 100% 正常）=================
plt.rcParams["font.family"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ================= 读取 Excel =================
def read_excel(file):
    sheet_data = {}
    xl = pd.ExcelFile(file)
    sheet_names = xl.sheet_names

    for idx, name in enumerate(sheet_names, 1):
        print(f"[{idx}/{len(sheet_names)}] 读取工作表：{name}")
        df = pd.read_excel(file, sheet_name=name, engine="openpyxl")
        sheet_data[name] = df.to_numpy()

    return sheet_names, sheet_data

# ================= 按坐标绘制高程热力图 =================
def plot_elevation_heatmap(sheet_names_list, sheet_data):
    plt.figure(figsize=(12, 8))

    # 遍历所有工作表，叠加绘制
    for name in sheet_names_list:
        data = sheet_data[name]
        try:
            # 【自动识别列：经度 | 纬度 | 高程】
            lon = data[:, 0]   # 经度 X
            lat = data[:, 1]   # 纬度 Y
            ele = data[:, 2]   # 高程 Z

            # 绘制高程热力散点图
            sc = plt.scatter(
                lon, lat,      # 坐标
                c=ele,         # 颜色 = 高程
                cmap="terrain",# 地形配色
                s=30,          # 点大小
                alpha=0.8,     # 透明度
                label=name
            )
        except Exception as e:
            print(f"⚠️ {name} 格式异常，跳过：{e}")

    # 图表样式
    plt.colorbar(sc, label="高程（米）")  # 高程颜色条
    plt.xlabel("经度（Longitude）", fontsize=12)
    plt.ylabel("纬度（Latitude）", fontsize=12)
    plt.title("秦直道及周边地形 高程热力图", fontsize=16)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# ================= 主程序 =================
if __name__ == "__main__":
    file = '附件2  秦直道及周边地形和相关遗迹的数据_副本.xlsx'
    sheet_names_list, sheet_data = read_excel(file)
    plot_elevation_heatmap(sheet_names_list, sheet_data)