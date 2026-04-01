import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

# ====================== 中文显示设置（Mac 100% 生效）======================
plt.rcParams["font.family"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ====================== 读取 Excel ======================
def read_excel(file):
    sheet_data = {}
    xl = pd.ExcelFile(file)
    sheet_names = xl.sheet_names

    for idx, name in enumerate(sheet_names, 1):
        print(f"[{idx}/{len(sheet_names)}] 读取工作表：{name}")
        df = pd.read_excel(file, sheet_name=name, engine="openpyxl")
        sheet_data[name] = df.to_numpy()  # 转成 numpy 数组

    return sheet_names, sheet_data

# ====================== 绘图 ======================
def generate_image(sheet_names_list, sheet_data):
    plt.figure(figsize=(10, 6))  # 设置画布大小
    
    for name in sheet_names_list:
        data = sheet_data[name]
        try:
            x = data[:, 0]  # 第1列
            y = data[:, 1]  # 第2列
            plt.plot(x, y, linewidth=2, label=name)
        except Exception as e:
            print(f"工作表 {name} 数据格式异常：{e}")

    # 中文标签（你可以自己改）
    plt.xlabel('X 轴', fontsize=12)
    plt.ylabel('Y 轴', fontsize=12)
    plt.title('秦直道及周边地形数据', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# ====================== 主程序 ======================
if __name__ == "__main__":
    file = '附件2  秦直道及周边地形和相关遗迹的数据_副本.xlsx'
    sheet_names_list, sheet_data = read_excel(file)
    generate_image(sheet_names_list, sheet_data)