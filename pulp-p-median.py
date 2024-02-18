from pulp import *
import numpy as np
import geopandas as gp
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import pandas as pd

def p_median_algorithm(SouthCarolina_shp, P_RELOCATED=3):
    """
    P-Median algorithm implementation.

    Parameters:
        SouthCarolina_shp (geopandas.GeoDataFrame): GeoDataFrame containing shapefile data.
        P_RELOCATED (int): Number of facilities to be placed.

    Returns:
        rslt (list): List of indices of facility nodes.
        fac_loc (geopandas.GeoDataFrame): GeoDataFrame containing geometry of facility nodes.
    """
    # Create demand and facilities variables
    demand = np.arange(0, 46, 1)
    facilities = np.arange(0, 20, 1)

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
    prob += sum([X[j] for j in facilities]) == P_RELOCATED
    for i in demand:
        prob += sum(Y[i][j] for j in facilities) == 1
    for i in demand:
        for j in facilities:
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
#read a sample shapefile SC一共46个县
SouthCarolina_shp = gp.read_file("./gadm41_USA_shp/gadm41_USA_SC_countries.shp") 
# 读取 CSV 文件 
# https://rfa.sc.gov/data-research/population-demographics/census-state-data-center/population-data/population-estimates-counties
csv_data = pd.read_csv('./popestbycounty1020.csv')
# 合并到 SouthCarolina_shp
SouthCarolina_shp['NAME_2'] = SouthCarolina_shp['NAME_2'].str.upper()
SouthCarolina_shp = pd.merge(SouthCarolina_shp, csv_data, left_on='NAME_2', right_on='COUNTY', how='left')

# 调用函数并绘制结果
rslt, fac_loc = p_median_algorithm(SouthCarolina_shp)
fig, ax = plt.subplots(figsize=(5, 5))
SouthCarolina_shp.centroid.plot(ax=ax, markersize=SouthCarolina_shp['2020ESTIMATES'] / 1000)
fac_loc.centroid.plot(ax=ax, color="red", markersize=300)
plt.show()
