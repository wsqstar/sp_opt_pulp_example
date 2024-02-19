from pulp import *
import numpy as np
import geopandas as gp
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import pandas as pd

def p_median_algorithm(SouthCarolina_shp, P_LOCATED=3,fixed_facilities=[]):

    """
    P-Median algorithm implementation.

    Parameters:
        SouthCarolina_shp (geopandas.GeoDataFrame): GeoDataFrame containing shapefile data.
        P_LOCATED (int): Number of facilities to be placed.
        fixed_facilities (list): List of indices of facilities to be fixed.

    Returns:
        rslt (list): List of indices of facility nodes.
        fac_loc (geopandas.GeoDataFrame): GeoDataFrame containing geometry of facility nodes.
    """
    # Create demand and facilities variables
    demand = np.arange(0, 46, 1)
    facilities = np.arange(0, 46, 1)

    # Calculate distance matrix
    coords = list(zip(SouthCarolina_shp.centroid.x, SouthCarolina_shp.centroid.y))
    d = cdist(coords, coords)

    # Demand for each county
    h = SouthCarolina_shp['2020ESTIMATES'].values

    # Declare facilities variables
    X = LpVariable.dicts('X_%s', (facilities), cat='Binary')

    # Declare demand-facility pair variables
    Y = LpVariable.dicts('Y_%s_%s', (demand, facilities), cat='Binary')

    prob = LpProblem('P_Median_GA', LpMinimize)

    # Objective function
    prob += sum(sum(h[i] * d[i][j] * Y[i][j] for j in facilities) for i in demand)

    # Constraints
    prob += sum([X[j] for j in facilities]) == P_LOCATED
    for i in demand:
        prob += sum(Y[i][j] for j in facilities) == 1
    for i in demand:
        for j in facilities:
            if j in fixed_facilities:
                prob += Y[i][j] == 1  # 将固定的设施的分配变量设为1
            else:
                prob += Y[i][j] <= X[j]

    # Solve the problem
    prob.solve()

    print("Status:", LpStatus[prob.status])
    print("Objective: ", value(prob.objective))

    # Get facility nodes
    rslt = []
    for v in prob.variables():
        subV = v.name.split('_')
        if subV[0] == "X" and v.varValue == 1:
            rslt.append(int(subV[1]))

    # Get the geometry of the facility nodes
    fac_loc = SouthCarolina_shp.iloc[rslt, :]

    return rslt, fac_loc


def main():
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
    # P_LOCATED是重新选址的数目 fixed_facilities = [0, 1, 2] (例)的数量
    TOTAL_FACILITIES = 6
    fixed_facilities = [0, 1, 2]  # 例如，固定前三个设施不动
    P_LOCATED= TOTAL_FACILITIES - len(fixed_facilities)
    rslt, fac_loc = p_median_algorithm(SouthCarolina_shp,P_LOCATED,fixed_facilities)
    print(f'Serial number of site selection {rslt}' )
    #Step3 可视化
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(f'Serial number of site selection {rslt}')  # 修改标题
    SouthCarolina_shp.centroid.plot(ax=ax, markersize=SouthCarolina_shp['2020ESTIMATES'] / 1000)
    fac_loc.centroid.plot(ax=ax, color="red", markersize=300,marker="*", label="New Facilities")
    # 显示被固定设施的位置
    fixed_facility_locs = SouthCarolina_shp.iloc[fixed_facilities]
    fixed_facility_locs.centroid.plot(ax=ax, color="blue", markersize=300, marker="*", label="Fixed Facilities")
    plt.legend()
    plt.show()

# 当直接执行这个脚本文件时，main 函数将会被执行
if __name__ == "__main__":
    main()
