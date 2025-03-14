from pyomo.environ import ConcreteModel, Var, Objective, Constraint, NonNegativeReals, minimize, SolverFactory
import matplotlib.pyplot as plt
import numpy as np

# Data / Parameters
load = [99, 93, 88, 87, 87, 88, 109, 127, 140, 142, 142, 140, 140, 140, 137, 139, 146, 148, 148, 142, 134, 123, 108, 93]
lf_pv = [0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00, 9.80E-04, 2.47E-02, 9.51E-02, 1.50E-01, 2.29E-01, 2.98E-01, 3.52E-01,
         4.15E-01, 4.58E-01, 3.73E-01, 2.60E-01, 2.19E-01, 1.99E-01, 8.80E-02, 7.03E-02, 3.90E-02, 9.92E-03, 1.39E-06, 0.00E+00, 0.00E+00]
timestep = len(load)
c_pv = 2500  # €/kW
c_batt = 1000  # €/kWh
eff_batt_in = 0.95
eff_batt_out = 0.95
chargetime = 4  # heures pour charger complètement la batterie

# Model
model = ConcreteModel()

# Define model variables
model.pv_size = Var(domain=NonNegativeReals)  # Taille du champ PV (kW)
model.batt_size = Var(domain=NonNegativeReals)  # Capacité de la batterie (kWh)

model.batt_charge = Var(range(timestep), domain=NonNegativeReals)  # Charge de la batterie à chaque pas de temps (kWh)
model.batt_discharge = Var(range(timestep), domain=NonNegativeReals)  # Décharge de la batterie (kWh)
model.soc = Var(range(timestep), domain=NonNegativeReals)  # État de charge de la batterie (kWh)

# Constraint: Battery State of Charge (SOC) evolution
def soc_rule(model, t):
    if t == 0:
        return model.soc[t] == 500
    else:
        return model.soc[t] == model.soc[t-1] + eff_batt_in * model.batt_charge[t] - model.batt_discharge[t] / eff_batt_out

model.soc_con = Constraint(range(timestep), rule=soc_rule)

# Constraint: SOC must not exceed battery capacity
def soc_limit_rule(model, t):
    return model.soc[t] <= model.batt_size

model.soc_limit_con = Constraint(range(timestep), rule=soc_limit_rule)

#Constraint: SOC is the same at the end of the day as at the beginning
def soc_end_rule(model):
    return model.soc[timestep-1] == model.soc[0]
model.soc_end_con = Constraint(rule=soc_end_rule)

# Constraint: Charge and discharge must be lower than a fraction of battery capacity
def charge_limit_rule(model, t):
    return model.batt_charge[t] <= model.batt_size / chargetime

def discharge_limit_rule(model, t):
    return model.batt_discharge[t] <= model.batt_size / chargetime

model.charge_limit_con = Constraint(range(timestep), rule=charge_limit_rule)
model.discharge_limit_con = Constraint(range(timestep), rule=discharge_limit_rule)

# Constraint: Energy balance equation (production + batterie = consommation)
def energy_balance_rule(model, t):
    return load[t] == model.pv_size * lf_pv[t] + model.batt_discharge[t] - model.batt_charge[t]

model.energy_balance_con = Constraint(range(timestep), rule=energy_balance_rule)

# Define the objective function: Minimize investment cost (PV + battery)
def obj_rule(model):
    return c_batt * model.batt_size + c_pv * model.pv_size

model.obj = Objective(rule=obj_rule, sense=minimize)

# Solve the model
solver = SolverFactory('gurobi')
results = solver.solve(model, tee=True)

# Check solver status
if (results.solver.status == 'ok') and (results.solver.termination_condition == 'optimal'):
    print('Optimal PV size: ', model.pv_size.value, "kW")
    print('Optimal battery capacity: ', model.batt_size.value, "kWh")

    # Plotting results
    plt.figure(figsize=(10, 6))
    plt.plot(range(timestep), load, label='Load (kW)')
    plt.plot(range(timestep), np.array(lf_pv) * model.pv_size.value, label='PV production (kW)')
    plt.plot(range(timestep), [model.soc[t].value for t in range(timestep)], label='SOC (kWh)')

    plt.xlabel('Time step')
    plt.ylabel('Energy (kWh)')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Solver did not find an optimal solution.")
