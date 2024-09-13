import numpy as np
from pulp import *
import math

# Set Parameters
scenarios = [1, 2]  # 1: Prosperity, 2: Recession
invest = [1, 2]  # 1: Buy Stock, 2: Buy Bond
pScenario = [0.5, 0.5]  # Probability of each scenario
ReturnOfStock = [1.25, 1.06]  # Index[0]: Rate of return for stock in Prosperity, Index[1]: Rate of return for stock in Recession
ReturnOfBond = [1.14, 1.12] # Index[0]: Rate of return for bonds in Prosperity, Index[1]: Rate of return for bonds in Recession
goal = 80000
InitialInvest = 55000
ShortagePenalty = 4
SurplusReward = 1

# Extract h and T
# Master Problem
mlp = LpProblem('MasterLinearProblem', sense=LpMinimize)
x1 = LpVariable.dicts("x1", invest, lowBound=0, cat="continuous")
theta = LpVariable.dicts('Theta', [s for s in scenarios], cat="continuous")
mlp += lpSum(theta[s] for s in scenarios)
mlp += lpSum(x1[i] for i in invest) == InitialInvest

# Sub problems
h = {}
T = {}
for s1 in scenarios:
    sbp = LpProblem('SubProblem', sense=LpMinimize)
    y = LpVariable.dicts("y", [(s1) for s1 in scenarios], lowBound=0, cat="continuous")
    w = LpVariable.dicts("w", [(s1) for s1 in scenarios], lowBound=0, cat="continuous")

    sbp += (SurplusReward * y[s1]) - (ShortagePenalty * w[s1])

    sbp += ReturnOfStock[s1 - 1] * x1[1] + ReturnOfBond[s1 - 1] * x1[2] == goal + y[s1] - w[s1]

    h_tmp = []
    for name, constraint in sbp.constraints.items():
        h_tmp.append(-constraint.toDict()['constant'])
    h[s1] = h_tmp

    T_tmp = []
    for name, constraint in sbp.constraints.items():
        coeef_row = []
        for var in mlp.variables():
            if var != theta[1] and var != theta[2]:
                coeff = constraint.get(var, 0)
                coeef_row.append(coeff)
        T_tmp.append(coeef_row)
        T[s1] = T_tmp

# Implementing Algorithm ...
# L-shape method for 2-stage financial planning stochastic problem:
# Step 0: Set parameters (r, s, v)
r = 0
v = 0
s = 0

# Step 1: solve Master Linear Problem
mlp = LpProblem('MasterLinearProblem', sense=LpMinimize)
x1 = LpVariable.dicts("x1", invest, lowBound=0, cat="continuous")
theta = LpVariable.dicts('Theta', [s for s in scenarios], cat="continuous")
mlp += lpSum(theta[s] for s in scenarios)
mlp += lpSum(x1[i] for i in invest) == InitialInvest

meet_stop_constraint = False
while not meet_stop_constraint:
    v += 1
    print(f"Iteration {v} - step 1")
    mlp.solve(PULP_CBC_CMD(msg=False, keepFiles=True))
    print("objective: ", value(mlp.objective))
    if v == 1:
        theta_v = -math.inf
    else:
        theta_v = value(mlp.objective)

    print(f"Theta iteration {v} is {theta_v}")

    # Step 2: check that step 1 solution available on K2 or not.
    print(f"iteration {v} - step 2:")
    SimplexMultiplier = {}
    d = []
    D = []
    for s1 in scenarios:
        back_to_mlp = False
        fc = LpProblem('FeasibilityCut', sense=LpMinimize)
        feasibilityCut = [1]
        v_pos = LpVariable.dicts('v_pos', feasibilityCut, lowBound=0, cat="continuous")
        v_neg = LpVariable.dicts('v_neg', feasibilityCut, lowBound=0, cat="continuous")
        y = LpVariable.dicts("y", [(s1) for s1 in scenarios], lowBound=0, cat="continuous")
        w = LpVariable.dicts("w", [(s1) for s1 in scenarios], lowBound=0, cat="continuous")
        fc += lpSum(v_pos[i] + v_neg[j] for i in feasibilityCut for j in feasibilityCut)

        fc += ReturnOfStock[s1 - 1] * x1[1].varValue + ReturnOfBond[s1 - 1] * x1[2].varValue == goal + y[s1] - w[s1]

        fc.solve(PULP_CBC_CMD(msg=False, keepFiles=False))

        if value(fc.objective) > 0: # Adding feasibility cut to master problem.
            back_to_mlp = True
            r += 1

            pi = []
            for _, c in sbp.constraints.items():
                pi.append(c.pi)
            simplex_multiplier[s1] = pi

            # Feasibility 
            d.append(np.array([simplex_multiplier[s1]]) @ np.array(h[s1]))
            D.append(np.array([simplex_multiplier[s1]]) @ np.array(T[s1]))
            mlp.addConstraint(list(D @ np.array([x1[1], x1[2]]))[0] >= d[0])
            print('added faesibility cut to step 1.')

        print(f"Objective function value for scenario {s1} = {value(fc.objective)}")

        if back_to_mlp: # We want to return to the beginning of the while loop.
            continue

    # Step 3: solve SubProblem based on each scenario (K= 1, 2)
    print(f"Iteration {v} step 3")
    E = []
    e = []
    simplex_multiplier = {}
    for s1 in scenarios:
        sbp = LpProblem('SubProblem', sense=LpMinimize)
        y = LpVariable.dicts("y", [(s1) for s1 in scenarios], lowBound=0, cat="continuous")
        w = LpVariable.dicts("w", [(s1) for s1 in scenarios], lowBound=0, cat="continuous")

        sbp += (SurplusReward * y[s1]) - (ShortagePenalty * w[s1])
        sbp += ReturnOfStock[s1 - 1] * x1[1].varValue + ReturnOfBond[s1 - 1] * x1[2].varValue == goal + y[s1] - w[s1]

        sbp.solve(PULP_CBC_CMD(msg=False, keepFiles=False))

        pi = []
        for _, c in sbp.constraints.items():
            pi.append(c.pi)
        simplex_multiplier[s1] = pi

        e.append(pScenario[s1-1] * np.array([simplex_multiplier[s1]]) @ np.array(h[s1]))

        E.append(pScenario[s1-1] * np.array([simplex_multiplier[s1]]) @ np.array(T[s1]))

    e_sum = sum(e)
    E_sum = sum(E)
    w = e_sum - E_sum @ np.array([x1[1].varValue, x1[2].varValue])
    if w > theta_v:
        s = s + 1
        mlp.addConstraint(list(E_sum @ np.array([x1[1], x1[2]]))[0] + theta >= e_sum[0])
        print("added optima1ity cut to step 1.")
        print('***'*15)
    else:
        print('meeting stop condition!')
        print('***'*15)
        print(f"Master Problem Objective Function: {value(mlp.objective)}")
        print(f"x1: {x1[1].varValue}, x2: {x1[2].varValue}")
        meet_stop_constraint = True
