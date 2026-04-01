import matplotlib.pyplot as plt
import pandas as pd

# ====================== 中文显示 ======================
plt.rcParams["font.family"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ====================== 读取 Excel 坐标 ======================
def read_points(file):
    # 读取第一个工作表
    sheet_dic={}
    for i in range(1,6):
        try:
            df = pd.read_excel(file, sheet_name=i, engine="openpyxl")
            print(f"成功读取工作表 {i}: {df.shape[0]} 行")
            sheet_dic[i] = df
        except Exception as e:
            print(f"⚠️ 读取工作表 {i} 失败：{e}")
    
    
    # # 自动取前两列为 X、Y 坐标
    # x = df.iloc[:, 0].values  # 第一列：X坐标
    # y = df.iloc[:, 1].values  # 第二列：Y坐标
    return sheet_dic

# ====================== 绘制点位图 ======================
def plot_points(sheet_dic):
    plt.figure(figsize=(10, 8))
    colors = ['red', 'blue', 'green', 'orange', 'purple']  # 不同工作表不同颜色
    sheet_names = list(sheet_dic.keys())
    for key in sheet_dic.keys():
        df = sheet_dic[key]
        color = colors[key - 1] if key <= len(colors) else 'black'
        x = df.iloc[:, 0].values  # 第一列：X坐标
        y = df.iloc[:, 1].values  # 第二列：Y坐标
        plt.scatter(x, y, c=f"{color}", s=15, alpha=0.8, label=f"工作表 {sheet_names[key-1]}")  # 使用列名作为标签
    # 绘制坐标点（红色圆点，大小适中）
    
    
    # 样式
    plt.xlabel("X 坐标 (m)")
    plt.ylabel("Y 坐标 (m)")
    plt.title("秦直道及周边遗迹点位图")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.axis("equal")  # 保持坐标比例不变形
    plt.tight_layout()
    plt.show()

# ====================== 主程序 ======================
if __name__ == "__main__":
    file = "附件2  秦直道及周边地形和相关遗迹的数据_副本.xlsx"
    sheet_dic = read_points(file)
    print(f"读取到 {len(sheet_dic)} 个工作表")
    plot_points(sheet_dic)