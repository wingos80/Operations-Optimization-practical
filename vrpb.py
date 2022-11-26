import gurobipy as grb
import numpy as np
import os
import pandas as pd

model = grb.Model(name='VRPB')

def firstFit(weight, n, c):
    # Initialize result (Count of bins)
    res = 0
    # Create an array to store remaining space in bins, there can be at most n bins
    bin_rem = [0]*n

    # Place items one by one
    for i in range(n):
        # Find the first bin that can accommodate weight[i]
        j = 0
        # Initialize minimum space left and index of best bin
        min = c + 1
        bi = 0

        for j in range(res):
            if (bin_rem[j] >= weight[i] and bin_rem[j] -
                                    weight[i] < min):
                bi = j
                min = bin_rem[j] - weight[i]
            
        # If no bin could accommodate weight[i], create a new bin
        if (min == c + 1):
            bin_rem[res] = c - weight[i]
            res += 1
        else: # Assign the item to best bin
            bin_rem[bi] -= weight[i]
            
    return res
    
def Get_K(L, B, D, Q):
    

    L_demands = []
    B_demands = []

    for node in L:
        L_demands.append(D[node])
    for node in B:
        B_demands.append(D[node])

    L_demands.sort(reverse=True)
    B_demands.sort(reverse=True)




    Kl = firstFit(L_demands, len(L_demands), Q)
    Kb = firstFit(B_demands, len(B_demands), Q)
    return Kl, Kb
"""
#####################################
### GENERAL LINEAR MODEL TEMPLATE ###
#####################################

### Create model
# test_model = grb.Model(name='test')

### Create variables
# x = test_model.addVar(name='x', vtype=grb.GRB.CONTINUOUS, lb=0)
# y = test_model.addVar(name='y', vtype=grb.GRB.CONTINUOUS, lb=0)

### Set objective
# obj_fn = 5*x + 4*y
# test_model.setObjective(obj_fn, grb.GRB.MINIMIZE)

### Add constraints
# c1 = test_model.addConstr(x+y >= 8, name='c1')
# c2 = test_model.addConstr(2*x+y >= 10, name='c2')
# c3 = test_model.addConstr(x+4*y >= 11, name='c3')

### Run and optimize model
# test_model.optimize()
# test_model.write('test.lp')

# print(f'Objective function value: {test_model.objVal}')

# for v in test_model.getVars():
#     print(f'{v.varName} = {v.x}')
"""

###################
### Import data ###
###################


# Get path to current folder
cwd = os.getcwd()

# Get all instances
full_list = os.listdir(cwd)

# instance name
excel_file = 'test.xlsx'

# Load data for this instance
dataframe = pd.read_excel(os.path.join(cwd,excel_file),sheet_name='Sheet1') # edges are basically the variables in a VRP for ONE TRUCK!!
# trucks = np.arange(0,2,1) # TODO need to figure out how to calculate minimum and maximum number of trucks needed
a = dataframe['X coord']
# print(f'{a}')


############################
### Create the variables ###
############################

# variables dicts
s = {}  # list of all binary variables for the use of paths between node i and j
xi = {}  # list of all binary variables for the use of path between nodes i in L and j in B0
l = {}   # list of continuous numbers indicating the amount of cargo transported between nodes i and j

# sets dicts
L = {}
B = {}
L0 = {}
B0 = {}
Cu = {}
V = {}

# parameters dicts
C = {}
D = {}

# calling the data from excel file
nodes = dataframe['Node number']
types = dataframe['Type']
x_coords = dataframe['X coord']
y_coords = dataframe['Y coord']
demands = dataframe['Demand']

# looping through all nodes and populating all variable, set, and parameters dictionaries
for node1 in nodes:
    node1_type = types[node1]
    node1_coords = np.array([x_coords[node1], y_coords[node1]])
    V[node1] = node1

    if node1_type == 'L':
        L[node1] = node1
        L0[node1] = node1
        Cu[node1] = node1
        D[node1] = demands[node1]

    elif node1_type == 'B':
        B[node1] = node1
        B0[node1] = node1
        Cu[node1] = node1
        D[node1] = demands[node1]

    elif node1_type == 'D':
        L0[node1] = node1
        B0[node1] = node1
    
    for node2 in nodes:
        node2_type = types[node2]
        node2_coords = np.array([x_coords[node2], y_coords[node2]])

        # populating s and l dicts
        s[node1,node2] = model.addVar(lb=0, ub=1, vtype=grb.GRB.BINARY, name=f's_{node1}_{node2}')
        l[node1,node2] = model.addVar(vtype=grb.GRB.CONTINUOUS, name=f'l_{node1}_{node2}') # maybe need to add lower bound = 0??
        C[node1,node2] = np.sqrt((x_coords[node2]- x_coords[node1])**2 + (y_coords[node2]- y_coords[node1])**2)
        # C[node1,node2] = np.linalg.norm(node2_coords - node1_coords)
            
        # populate xi dict
        if node1_type == 'L' and (node2_type == 'B' or node2_type == 'D') :
            xi[node1,node2] = model.addVar(lb=0, ub=1, vtype=grb.GRB.BINARY, name=f'xi_{node1}_{node2}')

Q = 75

# solving the knapscak problem to calculate minimum Kl and Kb
Kl, Kb = Get_K(L, B, D, Q)
print(f' Kl Kb ar: {Kl} {Kb}')

print(f' these are all the variables.\n s:\n{s}\n\n xi:\n{xi}\n\n l:\n{l}\n\n')

print(f'these are the demands: {D}')
print(f'these are the types: {types}')

model.update()

##############################
### Create the constraints ###
##############################
model.addConstr(grb.quicksum(s[i,j] for i in L0 for j in L) == len(L), name='c1') # C1  

for j in L:
    model.addConstr(grb.quicksum(l[i,j] for i in L0 if i!=j) - grb.quicksum(l[j,k] for k in L0 if k !=j) == D[j], name=f'c2_{j}') #C2   
    
    model.addConstr(grb.quicksum(s[i,j] for i in L0) == 1, name=f'c3_{j}') # C3
    
    model.addConstr(grb.quicksum(s[j,k] for k in L) + grb.quicksum(xi[j,k] for k in B0) == grb.quicksum(s[i,j] for i in L0), name=f'c4_{j}') #C4
    
    for i in L0:
        model.addConstr(l[i,j] <= Q*s[i,j], name=f'c5_{i}{j}') # C5

model.addConstr(grb.quicksum(s[0,j] for j in L) >= Kl, name='c6') # C6

model.addConstr(grb.quicksum(s[i,j] for i in B for j in B0) == len(B), name='c7') # C7

for j in B:
    model.addConstr(grb.quicksum(l[i,j] for i in B if i!=j) - grb.quicksum(l[j,k] for k in B0 if k !=j) == -D[j], name=f'c8_{j}') #C8 

    # BUG HERE, show daniel why theres bug here
    model.addConstr(grb.quicksum(s[i,j] for i in B0) == 1, name=f'c9_{j}') # C9

    model.addConstr(grb.quicksum(s[k,j] for k in B) + grb.quicksum(xi[k,j] for k in L) == grb.quicksum(s[i,j] for i in B0), name=f'c10_{j}') #C10 

    for i in B0:
        model.addConstr(l[j,i] <= Q*s[j,i], name=f'c11_{j}{i}') # C11

model.addConstr(Kb <= grb.quicksum(s[i, 0] for i in B), name='c12.1') # C12.1
model.addConstr(grb.quicksum(s[i, 0] for i in B) <= grb.quicksum(s[0,j] for j in L), name='c12.2') # C12.2
        
model.addConstr(grb.quicksum(s[i,0] for i in B) + grb.quicksum(xi[i,0] for i in L) == grb.quicksum(s[0,j] for j in L), name='c13') # C13

for i in V:
    for j in V:
        model.addConstr(s[i,j]+s[j,i] <= 1, name=f'c14_{i}_{j}') # C14

model.addConstr(grb.quicksum(s[i,j] for i in B for j in L) == 0, name='c15') # C15
# model.addConstr(grb.quicksum(s[0,j] for j in B) == 0, name='c16') # C16
# model.addConstr(grb.quicksum(s[i,j] for i in L for j in B0) == 0, name='c17') # C17

model.update()

#####################
### Set objective ###
#####################


# obj = grb.quicksum(C[i,j]*s[i,j] for i in V for j in V) + grb.quicksum(C[i,j]*xi[i,j] for i in L for j in B0) + grb.quicksum(s[0,i] for i in V)
obj = grb.quicksum(C[i,j]*s[i,j] for i in V for j in V) + grb.quicksum(C[i,j]*xi[i,j] for i in L for j in B0)
model.setObjective(obj, grb.GRB.MINIMIZE)


model.update()

########################
## Optimize the model ##
########################

### Run and optimize model
model.optimize()
model.write('test.lp')

print(f'Objective function value: {model.objVal}')

for v in model.getVars():
    print(f'{v.varName} = {v.x}')