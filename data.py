import pandas as pd
import numpy as np

pp = np.arange(7)
dc = np.arange(5)
region = np.arange(8)
scenario = np.arange(7)


file_path = 'assignment2data.xlsx'

df_handling_costs = pd.read_excel(file_path, sheet_name='handling_costs', header=0, index_col=0)
df_cost_pp_dc = pd.read_excel(file_path, sheet_name='plant_to_dc', header=0, index_col=0)
df_cost_dc_region = pd.read_excel(file_path, sheet_name='dc_to_region', header=0, index_col=0)
df_demand = pd.read_excel(file_path, sheet_name='demand', header=0, index_col=0)

handling_costs = {}
cost_pp_dc = {}
cost_dc_region = {}
demand = {}

for d in dc:
    handling_costs = df_handling_costs.to_numpy()[0]
    for r in region:
        cost_dc_region[(d, r)] = df_cost_dc_region.iloc[d, r]
for p in pp:
    for d in dc:
        cost_pp_dc[(p, d)] = df_cost_pp_dc.iloc[p, d]
for w in scenario:
    for r in region:
        demand[(w, r)] = df_demand.iloc[w, r]

