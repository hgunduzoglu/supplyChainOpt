import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import data
import pandas as pd

model = pyo.ConcreteModel("Supply Chain Optimization")

# Sets
model.pp = pyo.Set(initialize=data.pp, doc='Set of Production Plants')
model.dc = pyo.Set(initialize=data.dc, doc='Set of Distribution Centers')
model.region = pyo.Set(initialize=data.region, doc='Set of Regions')
model.scenario = pyo.Set(initialize=data.scenario, doc='Set of Scenarios')

# Parameters
model.handling_costs = pyo.Param(model.dc, initialize=data.handling_costs, doc='Handling costs')
model.cost_pp_dc = pyo.Param(model.pp, model.dc, initialize=data.cost_pp_dc, doc='Cost of transporting from plants to distribution centers')
model.cost_dc_region = pyo.Param(model.dc, model.region, initialize=data.cost_dc_region, doc='Cost of transporting from distribution centers to regions')
model.demand = pyo.Param(model.scenario, model.region, initialize=data.demand, doc='Annual Demand per region per scenario')
model.costp = pyo.Param(initialize=1500000, doc='annual fixed operation cost of a Production Plant')
model.costd = pyo.Param(initialize=600000, doc='annual fixed operation cost of a Distribution Center')
model.capap = pyo.Param(initialize = 50000, doc='annual production capacity of a Production Plant')
model.capad = pyo.Param(initialize = 30000, doc='annual capacity of a Distribution Center')

# Variables
model.p = pyo.Var(model.pp, doc='1 if plant is open, 0 otherwise', within=pyo.Binary)
model.d = pyo.Var(model.dc, doc='1 if distribution center is open, 0 otherwise', within=pyo.Binary)
model.x = pyo.Var(model.scenario, model.pp, model.dc, doc='Amount transported from plant to distribution center in scenario', within=pyo.NonNegativeIntegers)
model.y = pyo.Var(model.scenario, model.dc, model.region, doc='Amount transported from distribution center to region in scenario', within=pyo.NonNegativeIntegers)
model.z = pyo.Var(within=pyo.NonNegativeReals) #New decision variable for robust optimization

# Constraints
def plant_capacity_rule(m, s, k):
    return sum(m.x[s, k, j] for j in m.dc) <= m.capap * m.p[k]
model.PlantCapacity = pyo.Constraint(model.scenario, model.pp, rule=plant_capacity_rule)
def dc_capacity_rule(m, s, j):
    return sum(m.x[s, k, j] for k in m.pp) <= m.capad * m.d[j]
model.DCCapacity = pyo.Constraint(model.scenario, model.dc, rule=dc_capacity_rule)
def flow_balance_rule(m, s, j):
    return sum(m.x[s, k, j] for k in m.pp) >= sum(m.y[s, j, i] for i in m.region)
model.FlowBalance = pyo.Constraint(model.scenario, model.dc, rule=flow_balance_rule)

def demand_satisfaction_rule(m, s, i):
    return sum(m.y[s, j, i] for j in m.dc) == m.demand[s, i]
model.DemandSatisfaction = pyo.Constraint(model.scenario, model.region, rule=demand_satisfaction_rule)



def robust_objective_constraint_rule(m, s):
    return m.z >= (
        # 1. Annual fixed operation costs
        sum(m.costp * m.p[k] for k in m.pp) +
        sum(m.costd * m.d[j] for j in m.dc) +

        # 2. DC handling costs
        sum(m.handling_costs[j] * sum(m.x[s, k, j] for k in m.pp) for j in m.dc) +

        # 3. PP → DC transportation costs
        sum(m.cost_pp_dc[k, j] * m.x[s, k, j] for k in m.pp for j in m.dc) +

        # 4. DC → Region transportation costs
        sum(m.cost_dc_region[j, i] * m.y[s, j, i] for j in m.dc for i in m.region)
    )
model.RobustObjectiveConstraint = pyo.Constraint(model.scenario, rule=robust_objective_constraint_rule)

# Objective
model.RobustObjective= pyo.Objective(expr=model.z, sense=pyo.minimize)

# Solver seçimi
solver = SolverFactory("gurobi")  

# Modeli çöz
result = solver.solve(model, tee=False)  


# 1. Açılan Üretim Tesisleri
plants_open = [(k + 1, pyo.value(model.p[k])) for k in model.pp]
df_plants = pd.DataFrame(plants_open, columns=["Plant", "Opened"])

# 2. Açılan Dağıtım Merkezleri
dcs_open = [(j + 1, pyo.value(model.d[j])) for j in model.dc]
df_dcs = pd.DataFrame(dcs_open, columns=["DC", "Opened"])

# 3. Plant to DC sevkiyatları (sadece >0 olanlar)
x_rows = []
for k in model.pp:
    for j in model.dc:
        row = {"Plant": k + 1, "DC": j + 1}
        total = 0
        for s in model.scenario:
            val = pyo.value(model.x[s, k, j])
            row[f"Scenario{s + 1}"] = val
            total += val
        if total > 0:
            x_rows.append(row)
df_x = pd.DataFrame(x_rows)

# 4. DC to Region sevkiyatları (sadece >0 olanlar)
y_rows = []
for j in model.dc:
    for i in model.region:
        row = {"DC": j + 1, "Region": i + 1}
        total = 0
        for s in model.scenario:
            val = pyo.value(model.y[s, j, i])
            row[f"Scenario{s + 1}"] = val
            total += val
        if total > 0:
            y_rows.append(row)
df_y = pd.DataFrame(y_rows)

# 5. Her senaryo için oluşan toplam maliyet
scenario_costs = []
for s in model.scenario:
    cost = (
        sum(pyo.value(model.costp) * pyo.value(model.p[k]) for k in model.pp) +
        sum(pyo.value(model.costd) * pyo.value(model.d[j]) for j in model.dc) +
        sum(pyo.value(model.handling_costs[j]) * sum(pyo.value(model.x[s, k, j]) for k in model.pp) for j in model.dc) +
        sum(pyo.value(model.cost_pp_dc[k, j]) * pyo.value(model.x[s, k, j]) for k in model.pp for j in model.dc) +
        sum(pyo.value(model.cost_dc_region[j, i]) * pyo.value(model.y[s, j, i]) for j in model.dc for i in model.region)
    )
    scenario_costs.append({"Scenario": s + 1, "TotalCost": cost})
df_costs = pd.DataFrame(scenario_costs)

# 6. Z değeri (amaç fonksiyonu)
z_value = pyo.value(model.z)

# Excel'e yaz
with pd.ExcelWriter("question2b_output.xlsx") as writer:
    df_plants.to_excel(writer, sheet_name="Plants", index=False)
    df_dcs.to_excel(writer, sheet_name="DCs", index=False)
    df_x.to_excel(writer, sheet_name="Plant_to_DC", index=False)
    df_y.to_excel(writer, sheet_name="DC_to_Region", index=False)
    df_costs.to_excel(writer, sheet_name="ScenarioCosts", index=False)
    pd.DataFrame([{"RobustObjective_z": z_value}]).to_excel(writer, sheet_name="Objective", index=False)

print("Excel output saved as 'question2b_output.xlsx'")

