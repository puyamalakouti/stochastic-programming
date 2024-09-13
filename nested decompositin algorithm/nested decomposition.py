import numpy as np
from pulp import *

# Defining Scenario Tree, Scenarios and probability
scenario_Tree = [[2, 2.4, -16], [2.071, 2.486, -16.571], [2.143, 2.571, -17.143], [2.214, 2.657, -17.714],
                 [2.286, 2.743, -18.286], [2.357, 2.829, -18.857], [2.429, 2.914, -19.429], [2.5, 3, -20],
                 [2.571, 3.086, -20.571], [2.643, 3.171, -21.143], [2.714, 3.257, -21.714], [2.786, 3.343, -22.286],
                 [2.857, 3.429, -22.857], [2.929, 3.514, -23.429], [3, 3.6, -24]]

# 15 scenarios
scenario = {1: 15, 2: 15}
T1 = np.zeros((scenario[1], 6, 3))
T2 = np.zeros((scenario[2], 4, 3))

total_scenario = scenario[1] * scenario[2] # 15*15

for i, scenario1 in enumerate(T1):
    T1[i][1][2] = 1
    T1[i][2][0] = scenario_Tree[i][0]
    T1[i][3][1] = scenario_Tree[i][1]
    T1[i][4][2] = scenario_Tree[i][2]

for i, scenario2 in enumerate(T2):
    T2[i][0][0] = scenario_Tree[i][0]
    T2[i][1][1] = scenario_Tree[i][1]
    T2[i][2][2] = scenario_Tree[i][2]

T = {1: T1, 2: T2} # Scenario Tree

# Create zero Matrices for second and third stages' probability and assign value to them in the next loop
p1 = 1
p2 = np.zeros(15)
p3 = np.zeros(15)

# Assign probability to each Stage {1:1 (Certainty) , 2:1/15, 3:1/225}
for index, i in enumerate(p2):
    p2[index] = 1 / 15
    p3[index] = 1 / total_scenario

# Create dictionary for probabilities
probability = {1: p1, 2: p2, 3: p3}

h3 = np.array([200, 240, 0, 6000])
h2 = np.array([500, 500, 200, 240, 0, 6000])
h = {2: h2, 3: h3}
coef_1 = np.array([150, 230, 260])
coef_2 = np.array([238, 210])
coef_3 = np.array([-170, -150, -36, -10])
H = 3 # Number of stages

generator_node = {2: 1, 3: 15}

pi = {2: [0] * scenario[1], 3: [0] * total_scenario} # Dual variables
sigma = {2: [0] * scenario[1], 3: [0]} # Dual Variables
constraint_num = {2: 6, 3: 4} # number of constraints at each stage

# Defining Decision variables
theta2 = LpVariable.dicts('theta2', [j for j in range(15)], cat="Continuous")
theta1 = LpVariable('theta1', cat="Continuous")
x1 = LpVariable.dicts('x1', [j for j in range(3)], 0, cat="Continuous")
x2 = LpVariable.dicts('x2', [(j, scenario1) for j in range(3) for scenario1 in range(scenario[1])], 0, cat="Continuous")
y2 = LpVariable.dicts('y2', [(j, scenario1) for j in range(2) for scenario1 in range(scenario[1])], 0, cat="Continuous")
w2 = LpVariable.dicts('w2', [(j, scenario1) for j in range(4) for scenario1 in range(scenario[1])], 0, cat="Continuous")
y3 = LpVariable.dicts('y3',
                      [(j, scenario1, scenario2) for j in range(2) for scenario1 in range(scenario[1]) for scenario2 in
                       range(scenario[2])], 0, cat="Continuous")
w3 = LpVariable.dicts('w3',
                      [(j, scenario1, scenario2) for j in range(4) for scenario1 in range(scenario[1]) for scenario2 in
                       range(scenario[2])], 0, cat="Continuous")


# Defining NLDS problems as a dictionary
# stage 1:
NLDS = {1: LpProblem(name=f"NLDS({1}-{1})", sense=LpMinimize), 2: [], 3: []}
NLDS[1] += lpSum([coef_1[j] * x1[j] for j in range(3)]) + theta1
NLDS[1] += lpSum([x1[j] for j in range(3)]) <= 500
NLDS[1] += (theta1 == 0, "first_iteration")
NLDS[1].solve()

# stage 2:
for k in range(scenario[1]):
    NLDS[2].append(LpProblem(f"NLDS({2}-{k + 1})", sense=LpMinimize))
    NLDS[2][k] += (lpSum([coef_1[j] * x2[j, k] for j in range(3)]) + lpSum([coef_2[j] * y2[j, k] for j in range(2)]) + lpSum([coef_3[j] * w2[j, k] for j in range(4)]) + theta2[k])
    NLDS[2][k] += lpSum([x2[j, k] for j in range(3)]) <= 500
    NLDS[2][k] += x2[2, k] <= 500 - x1[2].value()
    NLDS[2][k] += y2[0, k] - w2[0, k] >= 200 - T[1][k][2][0] * x1[0].value()
    NLDS[2][k] += y2[1, k] - w2[1, k] >= 240 - T[1][k][3][1] * x1[1].value()
    NLDS[2][k] += w2[2, k] + w2[3, k] <= -T[1][k][4][2] * x1[2].value()
    NLDS[2][k] += w2[2, k] <= 6000
    NLDS[2][k] += (theta2[k] == 0, "first_iteration")
    NLDS[2][k].solve()

# stage 3:
counter = 0
for scenario1 in range(scenario[1]):
    for scenario2 in range(scenario[2]):
        NLDS[3].append(LpProblem(f"NLDS({3}-{counter + 1})", sense=LpMinimize))
        NLDS[3][counter] += (lpSum([coef_2[j] * y3[j, scenario1, scenario2] for j in range(2)])
                             + lpSum([coef_3[j] * w3[j, scenario1, scenario2] for j in range(4)]))
        NLDS[3][counter] += y3[0, scenario1, scenario2] - w3[0, scenario1, scenario2] >= 200 - T[2][scenario2][0][0] * \
                            x2[0, scenario1].value()
        NLDS[3][counter] += y3[1, scenario1, scenario2] - w3[1, scenario1, scenario2] >= 240 - T[2][scenario2][1][1] * \
                            x2[1, scenario1].value()
        NLDS[3][counter] += w3[2, scenario1, scenario2] + w3[3, scenario1, scenario2] <= -T[2][scenario2][2][2] * x2[
            2, scenario1].value()
        NLDS[3][counter] += w3[2, scenario1, scenario2] <= 6000
        NLDS[3][counter].solve()
        counter += 1

# NLDS step1 function: store dual variables and update x and theta for next stage
def step1(NLDS, t, DIR, pi, sigma):
    x = []
    if t == 1:
        NLDS[t].solve()
        for k in range(scenario[1]):
            NLDS[t + 1][k].constraints["_C2"] = x2[2, k] <= 500 - x1[2].value()
            NLDS[t + 1][k].constraints["_C3"] = y2[0, k] - w2[0, k] >= 200 - T[t][k][2][0] * x1[0].value()
            NLDS[t + 1][k].constraints["_C4"] = y2[1, k] - w2[1, k] >= 240 - T[t][k][3][1] * x1[1].value()
            NLDS[t + 1][k].constraints["_C5"] = w2[2, k] + w2[3, k] <= - T[t][k][4][2] * x1[2].value()
        DIR = "FORE"
        if (t < H and DIR == "FORE"):
            t += 1
            return step1(NLDS, t, DIR, pi, sigma)
        else:
            return NLDS, t, DIR, pi, sigma
    if t == 2:
        for k in range(scenario[1]):
            NLDS[t][k].solve()
            for i in range(len(list(NLDS[t][k].constraints.items()))):
                x.append(list(NLDS[t][k].constraints.items())[i][1].pi)
            pi[t][k] = x[:6]
            sigma[t][k] = x[6:]
            x = []

        counter = 0
        for scenario1 in range(scenario[1]):
            for scenario2 in range(scenario[2]):
                NLDS[t + 1][counter].constraints["_C1"] = y3[0, scenario1, scenario2] - w3[
                    0, scenario1, scenario2] >= 200 - T[t][scenario2][0][0] * x2[0, scenario1].value()
                NLDS[t + 1][counter].constraints["_C2"] = y3[1, scenario1, scenario2] - w3[
                    1, scenario1, scenario2] >= 240 - T[t][scenario2][1][1] * x2[1, scenario1].value()
                NLDS[t + 1][counter].constraints["_C3"] = w3[2, scenario1, scenario2] + w3[3, scenario1, scenario2] <= - \
                T[t][scenario2][2][2] * x2[2, scenario1].value()
                counter += 1
        if t < H and DIR == "FORE":
            t += 1
            return step1(NLDS, t, DIR, pi, sigma)
        else:
            return NLDS, t, DIR, pi, sigma
    if t == 3:
        for k in range(total_scenario):
            NLDS[t][k].solve()
            for i in range(len(list(NLDS[t][k].constraints.items()))):
                x.append(list(NLDS[t][k].constraints.items())[i][1].pi)
            pi[t][k] = x[:]
            x = []
        DIR = "BACK"
        return NLDS, t, DIR, pi, sigma

# NLDS step2 function: optimality cuts and calculating E & e
def step2(NLDS, e1, e2, t, DIR, pi, sigma):
    cut_counter = 0
    if t == 1:
        t += 1
        step1(NLDS, t, DIR, pi, sigma)
    x_variable = []
    x_value = []

    if t == 3:
        counter = 0
        for k in range(generator_node[t]):
            auxiliary_E2 = np.zeros((scenario[2], 3))
            auxiliary_e2 = np.zeros(scenario[2])
            if len(e2[k]) == 0:
                x_variable.append(x2[0, k])
                x_variable.append(x2[1, k])
                x_variable.append(x2[2, k])
                del NLDS[t - 1][k].constraints["first_iteration"]
                # Calculating E and e
                for m in range(scenario[2]):
                    auxiliary_pi = np.array(pi[t][counter])
                    auxiliary_E2[m] = (((probability[t][m]) / (probability[t - 1][k])) * (auxiliary_pi @ T[t - 1][m]))
                    auxiliary_e2[m] = (((probability[t][m]) / (probability[t - 1][k])) * (auxiliary_pi @ h[t]))
                    counter += 1

                E2 = np.sum(auxiliary_E2, axis=0)
                auxiliary_e2_int = np.sum(auxiliary_e2, axis=0)
                # Adding optimality cut
                NLDS[t - 1][k] += lpDot(x_variable, E2) + theta2[k] >= auxiliary_e2_int
                x_variable = []
                e2[k].append(auxiliary_e2_int)
            else:
                x_variable.append(x2[0, k])
                x_variable.append(x2[1, k])
                x_variable.append(x2[2, k])
                x_value.append(x2[0, k].value())
                x_value.append(x2[1, k].value())
                x_value.append(x2[2, k].value())
                # Calculating E and e
                for m in range(scenario[2]):
                    auxiliary_pi = np.array(pi[t][counter])
                    auxiliary_E2[m] = (((probability[t][m]) / (probability[t - 1][k])) * (auxiliary_pi @ T[t - 1][m]))
                    auxiliary_e2[m] = (((probability[t][m]) / (probability[t - 1][k])) * (auxiliary_pi @ h[t]))
                    counter += 1

                E2 = np.sum(auxiliary_E2, axis=0)
                auxiliary_e2_int = np.sum(auxiliary_e2, axis=0)
                theta_hat = round((auxiliary_e2_int - E2 @ x_value), 2)
                x_value = []
                # Adding Optimality cut
                if theta_hat > theta2[k].value():
                    NLDS[t - 1][k] += lpDot(x_variable, E2) + theta2[k] >= auxiliary_e2_int
                    x_variable = []
                    e2[k].append(auxiliary_e2_int)
        t = t - 1
        return True, NLDS, e1, e2, t, DIR

    if t == 2:
        auxiliary_E1 = np.zeros((scenario[1], 3))
        auxiliary_e1 = np.zeros(scenario[1])
        if len(e1) == 0:
            x_variable.append(x1[0])
            x_variable.append(x1[1])
            x_variable.append(x1[2])
            del NLDS[t - 1].constraints["first_iteration"]
            for m in range(scenario[1]):
                auxiliary_pi = np.array(pi[t][m])
                auxiliary_sigma = np.array(sigma[t][m])
                e2_array = np.array(e2[m])
                auxiliary_E1[m] = (((probability[t][m]) / (probability[t - 1])) * (auxiliary_pi @ T[t - 1][m]))
                auxiliary_e1[m] = (((probability[t][m]) / (probability[t - 1])) * (
                            auxiliary_pi @ h[t] + auxiliary_sigma @ e2_array))

            E1 = np.sum(auxiliary_E1, axis=0)
            auxiliary_e1_int = np.sum(auxiliary_e1, axis=0)
            NLDS[t - 1] += lpDot(x_variable, E1) + theta1 >= auxiliary_e1_int
            x_variable = []
            e1.append(auxiliary_e1_int)
            cut_counter += 1
        else:
            x_variable.append(x1[0])
            x_variable.append(x1[1])
            x_variable.append(x1[2])
            x_value.append(x1[0].value())
            x_value.append(x1[1].value())
            x_value.append(x1[2].value())
            # Calculating E and e
            for m in range(scenario[1]):
                auxiliary_pi = np.array(pi[t][m])
                auxiliary_sigma = np.array(sigma[t][m])
                e2_array = np.array(e2[m])
                auxiliary_E1[m] = (((probability[t][m]) / (probability[t - 1])) * (auxiliary_pi @ T[t - 1][m]))
                auxiliary_e1[m] = (((probability[t][m]) / (probability[t - 1])) * (
                            auxiliary_pi @ h[t] + auxiliary_sigma @ e2_array))

            E1 = np.sum(auxiliary_E1, axis=0)
            auxiliary_e1_int = np.sum(auxiliary_e1)
            theta_hat = round((auxiliary_e1_int - E1 @ x_value), 2)
            x_value = []

            # Adding Optimality cuts
            if theta_hat > theta1.value():
                NLDS[t - 1] += lpDot(x_variable, E1) + theta1 >= auxiliary_e1_int
                x_variable = []
                e1.append(auxiliary_e1_int)
                cut_counter += 1

        # Checking termination condition
        if cut_counter == 0:
            return False, NLDS, e1, e2, t, DIR
        else:
            t = t - 1
            DTR = "FORE"
            return True, NLDS, e1, e2, t, DIR

DIR = "FORE"
iters = 0
t = 1
k = 0
e1 = []
e2 = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
j = True
x = []

# Continue till finding Optimal Solution
while j:
    if t == 1:
        iters += 1
    NLDS, t, DIR, pi, sigma = step1(NLDS, t, DIR, pi, sigma)
    j, NLDS, e1, e2, t, DIR = step2(NLDS, e1, e2, t, DIR, pi, sigma)

print(f"Nested Decomposition algorithm solved by {iters} iterations.")
print("Stage 1: ")
print("NLDS(1,1)")
print("Objective Function: ", value(NLDS[1].objective))
print("ِDecision Variables: ")
for v in NLDS[1].variables():
    print(f"{v.name}: {round(v.value(), 3)}")
print("*" * 100)

print("Stage 2: ")
for i in range(scenario[1]):
    print(f"NLDS({2}-{i + 1})")
    print("Objective Function: ", value(NLDS[2][i].objective))
    print("ِDecision Variables: ")
    for v in NLDS[2][i].variables():
        print(f"{v.name}: {round(v.value(), 3)}")
print("*" * 100)

print("Stage 3: ")
for i in range(total_scenario):
    print(f"NLDS({3}-{i + 1})")
    print("Objective Function: ", value(NLDS[3][i].objective))
    print("ِDecision Variables: ")
    for v in NLDS[3][i].variables():
        print(f"{v.name}: {round(v.value(), 3)}")
        