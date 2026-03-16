import pandas as pd 
import numpy as np
import math
from scipy.ndimage import convolve

import os
import pandas as pd

def is_numeric_string(s):
    """辅助函数：判断字符串是否可转为数字（用于坐标校验）"""
    print("is_numeric_string")
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

def extract_coords_from_df(df):
    """辅助函数：从DataFrame提取x/y坐标（列=X，索引=Y），校验数字格式"""
    # 空DataFrame直接返回None
    print("extract_coords_from_df")
    if df.empty:
        print("  → 数据为空！")
        return None, None
    print(df.columns)
    
    # 1. 提取列名作为X坐标（过滤非数字）
    x_cols = [col for col in df.columns if is_numeric_string(col)]
    if len(x_cols) == 0:
        print("  → 无有效X坐标（列名非数字）！")
        return None, None
    x_codes = np.array([float(col) for col in x_cols])
    print(df.index)
    # 2. 提取索引作为Y坐标（过滤非数字）
    y_index = [idx for idx in df.index if is_numeric_string(idx)]
    if len(y_index) == 0:
        print("  → 无有效Y坐标（索引非数字）！")
        return None, None
    y_codes = np.array([float(idx) for idx in y_index])
    
    return x_codes, y_codes


    
def read_all_sheets_and_coords(file_path):
    """读取文件并转换成X/Y坐标"""
    print("read_all_sheets_and_coords")
    all_results = {}
    if not os.path.exists(file_path):
        print(f"错误：文件 {os.path.basename(file_path)} 不存在！")
        return all_results
    
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    try:
        if ext == ".csv":
            # CSV读取：取消index_col=0，识别表头
            try:
                df = pd.read_csv(file_path, encoding="gbk")
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="utf-8-sig")
            sheet_name = "CSV_数据"
            x, y = extract_coords_from_df(df)
            if x is not None and y is not None:
                all_results[sheet_name] = (x, y, df.values.astype(np.float32))
        
        elif ext == ".xlsx":
            xl_file = pd.ExcelFile(file_path, engine="openpyxl")
            sheet_names = xl_file.sheet_names
            print(f"\n开始读取Excel（共{len(sheet_names)}个工作表）：")
            
            for sheet_name in sheet_names:
                print(f"\n处理工作表：{sheet_name}")
                # Excel读取：取消index_col=0，识别表头
                df = pd.read_excel(xl_file, sheet_name=sheet_name, engine="openpyxl")
                x, y = extract_coords_from_df(df)
                if x is not None and y is not None:
                    all_results[sheet_name] = (x, y, df.values.astype(np.float32))
        
        else:
            print(f"错误：不支持的文件类型 {ext}！")
    
    except Exception as e:
        print(f"读取失败：{str(e)}")
    
    return all_results


def read_list(csv_path):
    """读取高程CSV数据，返回x坐标列表、y坐标列表、高程矩阵"""
    print("read_list")
    df = pd.read_csv(csv_path, index_col=0)
    # 提取x坐标（列名）和y坐标（索引）
    x_coords = np.array([float(col) for col in df.columns])
    y_coords = np.array([float(idx) for idx in df.index])
    # 构建高程矩阵（替换NA为NaN）
    elevation_matrix = df.values.astype(np.float32)
    return x_coords, y_coords, elevation_matrix

def read_execl_list(excel):     #读取xlsx文件
    """主函数中读取Excel表格"""
    print("read_excel_list")
    df=pd.ExcelFile(excel)
    sheet_names = df.sheet_names 
    sheetnumbers=0    
    print("该Excel文件所有的工作表：")
    for i,name in enumerate(sheet_names, 1):
        print(f"{i}.{name}") 
        sheetnumbers+=1
        sheet_imput=True
    while sheet_imput:
        choice_sheet=input(f"选择你要计算的表格(全选请按0)0~{sheetnumbers}：")
        print(choice_sheet.isdigit())

        if not(choice_sheet.isdigit()) or int(choice_sheet)>sheetnumbers+2 :
            print("超出范围/不是数字，重新输入")
            continue
        else:
            sheet_imput=False   
        all_results={}
        if int(choice_sheet)==0:     #计算所有表单
            print("Reading>>>>")
            xl_file = pd.ExcelFile(excel, engine="openpyxl")
            sheet_names = xl_file.sheet_names
            print(f"\n开始读取Excel（共{len(sheet_names)}个工作表）：")
            
            # 遍历所有工作表
            for idx, sheet_name in enumerate(sheet_names, 1):
                print(f"\n[{idx}/{len(sheet_names)}] 处理工作表：{sheet_name}")
                # 读取当前工作表（index_col=0 与CSV格式对齐）
                df = pd.read_excel(
                    xl_file,
                    sheet_name=sheet_name,
                    engine="openpyxl"
                )
                print(df.columns)
                print(df.values)
                # 提取并校验坐标
                x_codes, y_codes = extract_coords_from_df(df)
                if x_codes is not None and y_codes is not None:
                    # 构建高程矩阵（处理NaN，不影响后续计算）
                    high_matrix = df.values.astype(np.float32)
                    all_results[sheet_name] = (x_codes, y_codes, high_matrix)
                    print(f"✅ 成功 → x坐标{len(x_codes)}个，y坐标{len(y_codes)}个")
                else:
                    print(f"❌ 无有效数字坐标（跳过该表）")
        else:
            df = pd.read_excel(
                        excel,
                        sheet_name=sheet_names[int(choice_sheet)-1],  # 核心：指定工作表名称
                        engine="openpyxl",  # .xlsx固定用，.xls用xlrd
                        header=0  
                        )
            print(f"读取成功{sheet_names[int(choice_sheet)-1]}")
            print(df.columns)
            print(df.values)

    return  all_results# 提前return，后续代码完全没执行
    
    
def find_target_next_index(code,target):
    """#找目标最近的坐标的索引"""
    print("find_target_next_index")
    return np.argmin(np.abs(code-target)) #返回最近的索引

def calaulate_podu_pourway(need_work,x,y):
    """#计算地形的坡度和倾斜方向"""
    #3*3卷积计算坡度 变化率[dz/dx] = ((c + 2f + i) - (a + 2d + g)) / 8
    print("calaulate_podu_pourway")
    ex=np.array([-1,0,1],[-2,0,2],[-1,0,1])/(8*x) #设置x核
    ey=np.array([-1,-2,-1],[0,0,0],[1,2,1])/(8*y) #设置y核
    
    #计算各方向梯度
    dx=convolve(need_work,ex,mode="nearst")    #若没有值将取周围值近似
    dy=convolve(need_work,ey,mode="nearst") 
    
    #计算坡度 
    """公式  rise_run = √ ([dz/dx]2 + [dz/dy]2)
             slope_degrees = ATAN ( √ ([dz/dx]2 + [dz/dy]2) ) * 57.29578
             """
    slope=mp.arctan(np.sqrt(dx**2+dy**2)*57.29579)  #直接去180/pi的值

    #计算坡向
    """不具有下坡方向的平坦区域将赋值为 -1,坡向数据集中每个像元的值都可指示出该像元的坡度朝向。"""
    """aspect = 57.29578 * atan2 ([dz/dy], -[dz/dx])"""
    aspect=57.29578*arctan2(dy,-dx)
    aspect=(aspect+360)%360 #转成360度
    return slope,aspect

#何以味
def press_any_key_to_continue():
    """按任意键（回车）继续"""
    input("\n按任意键继续（按回车确认）...")

#测试模块
def test1():
    """测试寻找最近的坐标"""
    cvs_path="陕甘八县的高程数据.csv"
    output="testresult1.xlsx"   #自行更改后面数字
    target_points = [
        ("秦直道", 1292176.07, 4105424.08),
        ("秦直道", 1315893.15, 4085747.84),
        ("秦直道", 1319911.01, 4065228.58),
        ("秦直道", 1334988.77, 4042973.91),
        ("秦直道", 1345509.95, 4025746.98),
        ("秦直道", 1373110.96, 3974301.37),
        ("烽火台", 1307404.10, 4094344.62),
        ("烽火台", 1359078.89, 4011143.96),
        ("关隘", 1374526.53, 3965855.59),
        ("关隘", 1362751.20, 3998089.80)
]                               #标明题一数据

    #测试代码 读取坐标
    print("Reading.....")
    x_coords, y_coords, elevation_matrix = read_list(cvs_path)

    #计算值
    print("caculating points ")
    results=[]
    for i,(point,x,y) in enumerate(target_points,0):
        print(f"{point},{x},{y}")
        x_idx = find_target_next_index(x_coords, x)
        y_idx = find_target_next_index(y_coords, y)
        print(f"{i},找到{point}最近的坐标：{x_idx},{y_idx}")
    press_any_key_to_continue()

def test2():
    """测试读取表格函数"""
    # 调用测试
    file_path = "附件2  秦直道及周边地形和相关遗迹的数据.xlsx"  # 替换为你的文件路径
    all_sheet_data = read_all_sheets_and_coords(file_path)

    # 打印结果
    if all_sheet_data:
        for sheet, (x, y, _) in all_sheet_data.items():
            print(f"\n{sheet} 坐标示例：")
            print(f"X坐标前3个：{x[:3]}")
            print(f"Y坐标前3个：{y[:3]}")
#计算地形特征
def calculate_features(x,y,x_codes,y_codes,matrix):   #x:目标的x坐标 y:目标的y坐标 x-codes:周围y_codes:周围matrix:二维图
    """计算地形特征"""
    x_index=find_target_next_index(x_codes,x)
    y_index=find_target_next_index(y_codes,y)

    #计算高程数据
    elevation=matrix[y_codes,x_codes]
    if np.isnan(elevation):             #检测是否为Nane
        raise ValueError(f"目标{x}{y}无有效值")
    # 计算实际距离比例 分辨率
    x_rel=np.mean(np.diff(x_codes))
    y_rel=np.mean(np.diff(y_codes))

    #计算坡度 坡向
    slope,aspect=calaulate_podu_pourway(matrix,x_rel,y_rel)
    slope=slope[y_index,x_index]
    aspect=aspect[y_index,x_index]

    #生成缺失数据 通过3x3范围补齐
    w_size=3
    h_w_size=w_size//2      #取窗口半值
    #处理边界
    x_start=max(0,x_index-h_w_size)
    x_end=min(matrix.shape[1],x_index+h_w_size)     #用shape获取行数避免坐标超出范围
    y_start=max(0,y_index-h_w_size)
    y_end=min(matrix.shape[0],y_index+h_w_size)
    window=matrix[y_start:y_end,x_start:x_end]
    window=window[~np.isnan(window)]            #删除Nane

    max_e=np.max(window)
    min_e=np.min(window)
    mean_e=np.mean(window)
    std_e=np.std(window)        #计算标准差，研究高度起伏

    e_m_high=max_e-min_e
    r_e=elevation-mean_e        #Re： 从零开始

    return {
        "x/m":x,
        "y/m":y,
        "高程/m":elevation,
        "坡度/度":slope,
        "坡向/度":aspect,
        "最高高程/m":max_e,
        "最低高程/m":min_e,
        "平均高度":mean_e,
        "极差/m":e_m_high,
        "粗糙度":std_e,
        "相对高程":r_e
        }

#主程序
def main():
    """主函数 没啥要解释的"""
    eight_count_seat_csv="陕甘八县的高程数据.csv"
    Qin_zhi_dao="附件2  秦直道及周边地形和相关遗迹的数据.xlsx"
    output_xlsx="result1.xlsx"
    target_points = [
        ("秦直道", 1292176.07, 4105424.08),
        ("秦直道", 1315893.15, 4085747.84),
        ("秦直道", 1319911.01, 4065228.58),
        ("秦直道", 1334988.77, 4042973.91),
        ("秦直道", 1345509.95, 4025746.98),
        ("秦直道", 1373110.96, 3974301.37),
        ("烽火台", 1307404.10, 4094344.62),
        ("烽火台", 1359078.89, 4011143.96),
        ("关隘", 1374526.53, 3965855.59),
        ("关隘", 1362751.20, 3998089.80)
        ]
    #读取数据
    x_code,y_code,e_matrix=read_list(eight_count_seat_csv)
    #qx,qy,qz=read_list(Qin_zhi_dao)
    print(f"{x_code},{x_code}")
    #press_any_key_to_continue()

    read_execl_list(Qin_zhi_dao)
#运行主程序
if __name__=="__main__":
    print("test2:")
    test2()
    print("test1:")
    test1()
    print("main:")
    main()


