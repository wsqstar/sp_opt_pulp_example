from pulp import *
import numpy as np
import geopandas as gp
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import pandas as pd

def p_median_algorithm(SouthCarolina_shp, RE_LOCATED, fixed_facilities=[]):
    """
    P-Median算法实现。

    Parameters:
        SouthCarolina_shp (geopandas.GeoDataFrame): 包含shapefile数据的GeoDataFrame。
        RE_LOCATED (int): 待放置设施数量。
        fixed_facilities (list): 要固定的设施的索引列表。

    Returns:
        rslt (list): 设施节点的索引列表。
        fac_loc (geopandas.GeoDataFrame): 包含设施节点几何信息的GeoDataFrame。
    """
    # 创建需求和设施变量
    demand = np.arange(0, 46, 1)
    facilities = np.arange(0, 46, 1)

    # 计算距离矩阵
    coords = list(zip(SouthCarolina_shp.centroid.x, SouthCarolina_shp.centroid.y))
    # 此处也可以重新给出其他的经纬度坐标
    d = cdist(coords, coords)

    # 每个县的需求
    h = SouthCarolina_shp['2020ESTIMATES'].values

    # 声明设施变量
    X = LpVariable.dicts('X_%s', (facilities), cat='Binary')

    # 声明需求-设施对变量
    Y = LpVariable.dicts('Y_%s_%s', (demand, facilities), cat='Binary')

    prob = LpProblem('P_Median_GA', LpMinimize)

    # 目标函数
    # prob += sum(h[i] * d[i][j] * Y[i][j] for i in demand for j in facilities if j in fixed_facilities)
    prob += sum(sum(h[i] * d[i][j] * Y[i][j] for j in facilities) for i in demand)

    TOTAL_FACILITIES = RE_LOCATED + len(fixed_facilities)
    # 约束条件
    # This constraint indicates we must place exactly p facilities
    # 这个约束表示我们必须放置恰好 p 个设施 
    prob += sum([X[j] for j in facilities]) == RE_LOCATED
    # This constraint implies that a demand node i can only be serviced by one facility
    # 这个约束意味着一个需求节点 i 只能被一个设施服务
    for i in demand:
        prob += sum(Y[i][j] for j in facilities) == 1
    # This constraint implies that that demand node i 
    # can be serviced by a facility at j only if there is a facility at j
    # 这个约束意味着需求节点 i 只能被设施 j 服务，当且仅当在设施 j 处有一个设施存在时
    # It implicitly removes situation when X[j] = 0 but Y[i][j] = 1 
    # (node i is served by j but there is no facility at j)
    # 这个约束隐含地防止了 X[j] = 0 但是 Y[i][j] = 1 的情况发生
    # (即节点 i 被设施 j 服务但是设施 j 不存在)
    for i in demand:
        for j in facilities:
            if j in fixed_facilities:
                prob += Y[i][j] == 1  # 将固定的设施的分配变量设为1
            else:
                prob += Y[i][j] <= X[j]

    # 求解问题
    prob.solve()

    print("状态:", LpStatus[prob.status])
    # The minimized total demand-distance. The unit is person * meter
    # 最小化的总需求距离。单位为人 * 米
    print("目标值: ", value(prob.objective))

    # 获取设施节点
    rslt = [j for j in facilities if X[j].varValue == 1]

    # 获取设施节点的几何信息
    fac_loc = SouthCarolina_shp.iloc[rslt, :]

    return rslt, fac_loc



def main(show_fig=True):
    # Step1 整理数据
    #read a sample shapefile SC一共46个县
    SouthCarolina_shp = gp.read_file("./gadm41_USA_shp/gadm41_USA_SC_countries.shp") 
    # 读取 CSV 文件 
    # https://rfa.sc.gov/data-research/population-demographics/census-state-data-center/population-data/population-estimates-counties
    csv_data = pd.read_csv('./popestbycounty1020.csv')
    # 合并到 SouthCarolina_shp
    SouthCarolina_shp['NAME_2'] = SouthCarolina_shp['NAME_2'].str.upper()
    SouthCarolina_shp = pd.merge(SouthCarolina_shp, csv_data, left_on='NAME_2', right_on='COUNTY', how='left')

    # Step2 调用pulp并求解
    # 调用函数并绘制结果
    # RE_LOCATED是重新选址的数目 fixed_facilities = [0, 1, 2] (例)的数量
    TOTAL_FACILITIES = 4
    fixed_facilities = [9]  # 例如，固定[0, 1, 19]设施不动
    RE_LOCATED= TOTAL_FACILITIES - len(fixed_facilities)
    rslt, fac_loc = p_median_algorithm(SouthCarolina_shp,RE_LOCATED,fixed_facilities)
    print(f'Serial number of site selection {rslt}' )
    #Step3 可视化
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(f'Serial number of site selection {rslt}')  # 修改标题
    SouthCarolina_shp.centroid.plot(ax=ax, markersize=SouthCarolina_shp['2020ESTIMATES'] / 1000)
    fac_loc.centroid.plot(ax=ax, color="red", markersize=300,marker="*", label="New Facilities")
    # 显示被固定设施的位置
    if fixed_facilities:
        fixed_facility_locs = SouthCarolina_shp.iloc[fixed_facilities]
        fixed_facility_locs.centroid.plot(ax=ax, color="blue", markersize=300, marker="*", label="Fixed Facilities")
    plt.legend()
    if show_fig:
        print('❗请查看图像，关闭窗口后程序将继续执行。')
        plt.show()
    else:
        plt.savefig('./saved_fig/Step 1 Location allocation facility_locations.png')  # 将图像保存为文件而不弹出窗口



    #Step 4
    print('------下面是2SO4SAI方法的第二步------')
    




# 当直接执行这个脚本文件时，main 函数将会被执行
if __name__ == "__main__":
    
    main(show_fig=True)
