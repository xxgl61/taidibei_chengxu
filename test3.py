import pandas as pd
import numpy as np
from scipy.ndimage import convolve
import math

def read_elevation_data(csv_path):
    """读取高程CSV数据，返回x坐标列表、y坐标列表、高程矩阵"""
    df = pd.read_csv(csv_path, index_col=0)
    # 提取x坐标（列名）和y坐标（索引）
    x_coords = np.array([float(col) for col in df.columns])
    y_coords = np.array([float(idx) for idx in df.index])
    # 构建高程矩阵（替换NA为NaN）
    elevation_matrix = df.values.astype(np.float32)
    return x_coords, y_coords, elevation_matrix

def find_nearest_index(coords, target):
    """找到目标坐标在坐标列表中的最近索引（处理坐标匹配）"""
    return np.argmin(np.abs(coords - target))

def calculate_slope_aspect(elevation_matrix, x_res, y_res):
    """计算坡度（°）和坡向（°）
    参考：https://desktop.arcgis.com/zh-cn/arcmap/10.3/tools/spatial-analyst-toolbox/how-slope-works.htm
    """
    # 3x3卷积核（用于计算梯度）
    dx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / (8 * x_res)
    dy_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / (8 * y_res)
    
    # 计算x和y方向梯度
    dx = convolve(elevation_matrix, dx_kernel, mode='nearest')
    dy = convolve(elevation_matrix, dy_kernel, mode='nearest')
    
    # 计算坡度（弧度转角度）
    slope = np.arctan(np.sqrt(dx**2 + dy**2)) * (180 / np.pi)
    
    # 计算坡向（0°-360°，正北为0°顺时针）
    aspect = np.arctan2(dy, dx) * (180 / np.pi)
    aspect = (aspect + 360) % 360  # 转换为0-360°
    return slope, aspect

def calculate_terrain_features(x_target, y_target, x_coords, y_coords, elevation_matrix):
    """计算单个目标位置的地形特征"""
    # 1. 找到目标位置的最近索引（处理坐标不精确匹配）
    x_idx = find_nearest_index(x_coords, x_target)
    y_idx = find_nearest_index(y_coords, y_target)
    
    # 2. 基础高程特征
    elevation = elevation_matrix[y_idx, x_idx]
    if np.isnan(elevation):
        raise ValueError(f"目标位置({x_target}, {y_target})无有效高程数据")
    
    # 3. 计算分辨率（假设x、y方向等距）
    x_res = np.mean(np.diff(x_coords))  # x方向平均分辨率
    y_res = np.mean(np.diff(y_coords))  # y方向平均分辨率
    
    # 4. 坡度和坡向
    slope_matrix, aspect_matrix = calculate_slope_aspect(elevation_matrix, x_res, y_res)
    slope = slope_matrix[y_idx, x_idx]
    aspect = aspect_matrix[y_idx, x_idx]
    
    # 5. 局部高程统计（3x3窗口）
    window_size = 3
    half_win = window_size // 2
    # 窗口边界处理
    y_start = max(0, y_idx - half_win)
    y_end = min(elevation_matrix.shape[0], y_idx + half_win + 1)
    x_start = max(0, x_idx - half_win)
    x_end = min(elevation_matrix.shape[1], x_idx + half_win + 1)
    window = elevation_matrix[y_start:y_end, x_start:x_end]
    window = window[~np.isnan(window)]  # 去除窗口内的NaN值
    
    if len(window) < 5:  # 窗口内有效数据不足时的默认值
        max_elevation = elevation
        min_elevation = elevation
        mean_elevation = elevation
        std_elevation = 0
    else:
        max_elevation = np.max(window)
        min_elevation = np.min(window)
        mean_elevation = np.mean(window)
        std_elevation = np.std(window)
    
    # 6. 关键衍生特征
    elevation_range = max_elevation - min_elevation  # 高程极差
    roughness = std_elevation  # 地表粗糙度（窗口高程标准差）
    relative_elevation = elevation - mean_elevation  # 相对高程（相对于局部均值）
    
    return {
        "x坐标/m": x_target,
        "y坐标/m": y_target,
        "高程/m": round(elevation, 2),
        "坡度/°": round(slope, 2),
        "坡向/°": round(aspect, 2),
        "局部最大高程/m": round(max_elevation, 2),
        "局部最小高程/m": round(min_elevation, 2),
        "局部平均高程/m": round(mean_elevation, 2),
        "高程极差/m": round(elevation_range, 2),
        "地表粗糙度/m": round(roughness, 2),
        "相对高程/m": round(relative_elevation, 2)
    }

def main():
    # 配置参数
    csv_path = "陕甘八县的高程数据.csv"  # 附件1的CSV路径
    output_path = "result1.xlsx"  # 输出结果路径
    # 表2要求的目标位置坐标（问题1指定需输出的位置）
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
    
    # 1. 读取高程数据
    print("正在读取高程数据...")
    x_coords, y_coords, elevation_matrix = read_elevation_data(csv_path)
    print(f"{x_coords},｜{y_coords},｜{elevation_matrix}")
    # 2. 计算每个目标位置的地形特征
    print("正在计算地形特征...")
    results = []
    for idx, (point_type, x, y) in enumerate(target_points, 1):
        try:
            features = calculate_terrain_features(x, y, x_coords, y_coords, elevation_matrix)
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
                "地表粗糙度/m": features["地表粗糙度/m"],
                "相对高程/m": features["相对高程/m"]
            }
            results.append(features)
            print(f"完成位置{idx}（{point_type}）的特征计算")
            print(f"位置{idx}（{point_type}，({x}, {y})）")
        except Exception as e:
            print(f"位置{idx}（{point_type}，({x}, {y})）计算失败：{str(e)}")
    
    # 3. 保存结果到Excel
    if results:
        df_result = pd.DataFrame(results)
        df_result.to_excel(output_path, index=False, engine="openpyxl")
        print(f"\n结果已保存到：{output_path}")
        
        # 打印表2要求的结果（控制台输出确认）
        print("\n表2要求的位置特征结果：")
        print(df_result.to_string(index=False))
    else:
        print("无有效计算结果，未生成Excel文件")

if __name__ == "__main__":
    main()
"""def read_all_sheets_and_coords(file_path):
    读取文件并转换成X/Y坐标
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
    
    return all_results"""