import pandas as pd 
import numpy as np
import math
from scipy.ndimage import convolve
import os

def is_numeric_string(s):
    """辅助函数：判断字符串是否可转为数字（用于坐标校验）"""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

def extract_coords_from_df(df):
    """辅助函数：从DataFrame提取x/y坐标（列=X，索引=Y），校验数字格式"""
    """从你的Excel DataFrame中提取x/y坐标（适配第一列x坐标）"""
    print("\n=== 当前工作表数据结构（前3行）===")
    print(df.head(3))
    print(f"=== 所有列名 ===")
    print(df.columns.tolist())

    # 1. 定位x坐标列（优先第一列，或含“x”关键词的列）
    # 方案：先尝试第一列，若无效则模糊匹配“x坐标”列
    x_col = None
    # 尝试第一列（你的x坐标在第一列）
    first_col_name = df.columns[0]
    # 检查第一列的前10个数据是否为数字（验证是否是x坐标列）
    first_col_sample = df[first_col_name].head(10)
    if any(is_numeric_string(val) for val in first_col_sample):
        x_col = first_col_name
        print(f"✅ 识别第一列为x坐标列：{x_col}")
    else:
        # 模糊匹配含“x”的列（备用，防止列顺序变化）
        for col in df.columns:
            if "x" in str(col).lower():
                x_col = col
                print(f"✅ 模糊匹配x坐标列：{x_col}")
                break
    
    if not x_col:
        print("❌ 未找到x坐标列！")
        return None, None

    # 2. 定位y坐标列（含“y”关键词的列）
    y_col = None
    for col in df.columns:
        if "y" in str(col).lower() and col != x_col:
            y_col = col
            print(f"✅ 识别y坐标列：{y_col}")
            break
    if not y_col:
        print("❌ 未找到y坐标列！")
        return None, None

    # 3. 提取并清洗x/y坐标数据（第一列的单元格数据）
    # 过滤空值和非数字
    df_clean = df.dropna(subset=[x_col, y_col])
    df_clean = df_clean[
        df_clean[x_col].apply(is_numeric_string) &
        df_clean[y_col].apply(is_numeric_string)
    ]

    if df_clean.empty:
        print("❌ 无有效x/y坐标数据（全是空值或非数字）")
        return None, None

    # 4. 转为numpy数组（第一列x坐标数据）
    x_codes = np.array(df_clean[x_col].astype(float))
    y_codes = np.array(df_clean[y_col].astype(float))
    print(f"✅ 成功提取坐标：x坐标{len(x_codes)}个，y坐标{len(y_codes)}个")
    print(f"📊 x坐标范围：{x_codes.min():.2f} ~ {x_codes.max():.2f}")
    print(f"📊 y坐标范围：{y_codes.min():.2f} ~ {y_codes.max():.2f}")
    return x_codes, y_codes

def read_csv_list(csv_path):
    """读取高程CSV数据，返回x坐标列表、y坐标列表、高程矩阵"""
    df = pd.read_csv(csv_path, index_col=0)
    # 提取x坐标（列名）和y坐标（索引）
    x_coords = np.array([float(col) for col in df.columns])
    y_coords = np.array([float(idx) for idx in df.index])
    # 构建高程矩阵（替换NA为NaN）
    elevation_matrix = df.values.astype(np.float32)
    return x_coords, y_coords, elevation_matrix

def read_excel_list(excel):     #读取xlsx文件
    """主函数中读取Excel表格"""
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
                if idx!=5:
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
                        xl_file,
                        sheet_name=sheet_name,
                        usecols=["x坐标/m", "y坐标/m"],
                        engine="openpyxl"
                        )
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
            sheet_name=sheet_names[int(choice_sheet)-1]
            x_codes, y_codes = extract_coords_from_df(df)
            if x_codes is not None and y_codes is not None:
                # 构建高程矩阵（处理NaN，不影响后续计算）
                high_matrix = df.values.astype(np.float32)
                all_results[sheet_name] = (x_codes, y_codes, high_matrix)
                print(f"✅ 成功 → x坐标{len(x_codes)}个，y坐标{len(y_codes)}个")
            else:
                print(f"❌ 无有效数字坐标（跳过该表）")
            print(f"读取成功{sheet_names[int(choice_sheet)-1]}")
            print(df.columns)
            print(df.values)
            

    return  all_results# 提前return，后续代码完全没执行
    
    
def find_target_next_index(code,target):
    """#找目标最近的坐标的索引"""
    return np.argmin(np.abs(code-target)) #返回最近的索引

def safe_match_2d_coord(x_coords, y_coords, target_x, target_y, tolerance=None):
    print(f"返回x_coords: {x_coords},len={len(x_coords)}")
    print(f"返回y_coords: {y_coords},len={len(y_coords)}")

    if len(x_coords) == 0 or len(y_coords) == 0:
        print("错误：坐标数组为空！")
        return None, None, None, None

    # 分别找x/y最近索引（关键修复）
    idx_x = find_target_next_index(x_coords, target_x)
    idx_y = find_target_next_index(y_coords, target_y)

    # 实际匹配点
    best_x = x_coords[idx_x]
    best_y = y_coords[idx_y]

    # 计算距离
    dist = np.sqrt((best_x-target_x)**2 + (best_y-target_y)**2)

    if tolerance is not None and dist > tolerance:
        return None, None, None, None

    return int(idx_x), int(idx_y), best_x, best_y
       

def calaulate_podu_pourway(need_work,x,y):
    """#计算地形的坡度和倾斜方向"""
    #3*3卷积计算坡度 变化率[dz/dx] = ((c + 2f + i) - (a + 2d + g)) / 8
    ex = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])/(8*x)
    ey = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])/(8*y)
    
    #计算各方向梯度
    dx=convolve(need_work,ex,mode="nearest")    #若没有值将取周围值近似
    dy=convolve(need_work,ey,mode="nearest") 
    
    #计算坡度 
    """公式  rise_run = √ ([dz/dx]2 + [dz/dy]2)
             slope_degrees = ATAN ( √ ([dz/dx]2 + [dz/dy]2) ) * 57.29578
             """
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    #计算坡向
    """不具有下坡方向的平坦区域将赋值为 -1,坡向数据集中每个像元的值都可指示出该像元的坡度朝向。"""
    """aspect = 57.29578 * atan2 ([dz/dy], -[dz/dx])"""
    aspect=57.29578*np.arctan2(dy,-dx)
    aspect=(aspect+360)%360 #转成360度
    return slope,aspect

#何以味
def press_any_key_to_continue():
    """按任意键（回车）继续"""
    input("\n按任意键继续（按回车确认）...")

#计算地形特征
def calculate_features(x, y, x_codes, y_codes, matrix):
    """计算单个目标位置的地形特征"""
    
    x_index, y_index, safe_x, safe_y = safe_match_2d_coord(x_codes, y_codes, x, y)

    elevation = matrix[y_index, x_index]
    if np.isnan(elevation):
        raise ValueError(f"目标{x},{y}无有效值")

    # 分辨率
    x_rel = np.mean(np.diff(x_codes))
    y_rel = np.mean(np.diff(y_codes))

    # 坡度坡向
    slope_map, aspect_map = calaulate_podu_pourway(matrix, x_rel, y_rel)
    slope = slope_map[y_index, x_index]
    aspect = aspect_map[y_index, x_index]

    # ===== 3x3窗口 =====
    w_size = 3
    h = w_size // 2

    x0 = max(0, x_index - h)
    x1 = min(matrix.shape[1], x_index + h + 1)
    y0 = max(0, y_index - h)
    y1 = min(matrix.shape[0], y_index + h + 1)

    window = matrix[y0:y1, x0:x1]
    window = window[~np.isnan(window)]

    if len(window) < 5:
        max_elevation = elevation
        min_elevation = elevation
        mean_elevation = elevation
        std_elevation = 0
    else:
        max_elevation = np.max(window)
        min_elevation = np.min(window)
        mean_elevation = np.mean(window)
        std_elevation = np.std(window)

    elevation_range = max_elevation - min_elevation
    r_e = elevation - mean_elevation

    return {
        "x坐标/m": x,
        "y坐标/m": y,
        "高程/m": elevation,
        "坡度/°": slope,
        "坡向/°": aspect,
        "局部最大高程/m": max_elevation,
        "局部最小高程/m": min_elevation,
        "局部平均高程/m": mean_elevation,
        "高程极差/m": elevation_range,
        "地表粗糙度": std_elevation,
        "相对高程/m": r_e
    }

#主程序
def main():
    """主函数 没啥要解释的"""
    eight_count_seat_csv="陕甘八县的高程数据.csv"
    Qin_zhi_dao="附件2  秦直道及周边地形和相关遗迹的数据.xlsx"
    output="result2.xlsx"
    all_results=read_excel_list(Qin_zhi_dao)
    print(all_results)
    target_points=all_results["秦直道"] #表2要求的5个位置
        
    #读取数据
    x_coords,y_coords,e_matrix=read_csv_list(eight_count_seat_csv)
    print(f"{x_coords},{y_coords},{e_matrix}")
    print("caculating points ")
    qin_x_coords, qin_y_coords, _ = all_results["秦直道"]
    target_points = []
    for x, y in zip(qin_x_coords, qin_y_coords):
        target_points.append(("秦直道", x, y))
    print(f"all_results: {all_results}")
    print("读取完成，开始计算特征...")

    #计算特征
    results = []
    for idx, (point_type, x, y) in enumerate(target_points, 1):
        try:
            features = calculate_features(x, y, x_coords, y_coords, e_matrix)
            features["序号"] = idx
            features["类型"] = point_type
            # 调整列顺序（匹配表1格式）
            features = {
                "序号": features["序号"],
                "类型": features["类型"],
                "x坐标/m": features["x坐标/m"],
                "y坐标/m": features["y坐标/m"],
                "高程/m": features["高程/m"],
                "坡度/°": features["坡度/°"],
                "坡向/°": features["坡向/°"],
                "局部最大高程/m": features["局部最大高程/m"],
                "局部最小高程/m": features["局部最小高程/m"],
                "局部平均高程/m": features["局部平均高程/m"],
                "高程极差/m": features["高程极差/m"],
                "地表粗糙度": features["地表粗糙度"],
                "相对高程/m": features["相对高程/m"]
            }
            results.append(features)
            print(f"完成位置{idx}（{point_type}）的特征计算")
        except Exception as e:
            print(f"位置{idx}（{point_type}，({x}, {y})）计算失败：{str(e)}")
    

    # 3. 保存结果到Excel
    if results:
        df_result = pd.DataFrame(results)
        df_result.to_excel(output, index=False, engine="openpyxl")
        print(f"\n结果已保存到：{output}")
        
        # 打印表2要求的结果（控制台输出确认）
        print("\n表2要求的位置特征结果：")
        print(df_result.to_string(index=False))
    else:
        print("无有效计算结果，未生成Excel文件")

#运行主程序
if __name__=="__main__":
    main()


