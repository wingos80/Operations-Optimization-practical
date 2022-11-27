# # Loading packages that are used in the code
# import numpy as np
# import os
# import pandas as pd
# import time
# import gurobipy as grb
# import pickle
# from copy import deepcopy


# ####################
# ### pandas stuff ###
# ####################

# # Get path to current folder
# cwd = os.getcwd()

# # Get all instances
# full_list           = os.listdir(cwd)

# # instance name
# instance_name = 'test.xlsx'

# ################################
# ### setting up the map ###
# ################################

# # Load data for this instance
# edges = pd.read_excel(os.path.join(cwd,instance_name),sheet_name='something') # edges are basically the variables in a VRP for ONE TRUCK!!
# trucks = np.arange(0,2,1)
# print(f'i have trucks: {trucks}')

# # Define dictionaries with subsets of nodes that precede/follow a specific node
# # N_plus[i] is the dictionary containing nodes that are reachable FROM node i
# # N_minus[i] is the dictionary containing nodes that lead TO node i
# N_plus  = {}
# N_minus = {}

# # Set of all nodes
# N = sorted(np.unique(list(edges['From'])+list(edges['To'])))
# print(f'N is: {N}')
# for n in N:
#     N_plus[n]  = [row.To for index,row in edges.iterrows() if row.From == n]
#     N_minus[n] = [row.From for index,row in edges.iterrows() if row.To == n]


# startTimeSetUp = time.time()
# model = grb.Model()

# #################
# ### VARIABLES ###
# #################
# all_var = {}
# for truck in trucks:
#     for i in range(0,len(edges)):
#         var_key = edges['From'][i],edges['To'][i],truck
#         all_var[var_key] = model.addVar(lb=0, ub=1, vtype=grb.GRB.BINARY,name=f"edge from node {edges['From'][i]} to {edges['To'][i]} for truck {truck}")

# model.update()

# #################
# ### OBJECTIVE ###
# #################

# obj_func = 0
# itr = 0
# for truck in trucks:
#     for j in range(0,len(edges)):
#         obj_func += edges['Distance'][j]*all_var[edges['From'][j],edges['To'][j],truck]
#         itr+=1




# print(itr)
# # a = edges['From'][0:1]
# print(f'obective is.. {len(obj_func)}')
# # print(f'edges distance is \n{a}')

def firstFit(weight, n, c):
     
    # Initialize result (Count of bins)
    res = 0
 
    # Create an array to store
    # remaining space in bins
    # there can be at most n bins
    bin_rem = [0]*n
 
    # Place items one by one
    for i in range(n):
         
        # Find the first bin that
        # can accommodate
        # weight[i]
        j = 0
         
        # Initialize minimum space
        # left and index
        # of best bin
        min = c + 1
        bi = 0
 
        for j in range(res):
            if (bin_rem[j] >= weight[i] and bin_rem[j] -
                                       weight[i] < min):
                bi = j
                min = bin_rem[j] - weight[i]
             
        # If no bin could accommodate weight[i],
        # create a new bin
        if (min == c + 1):
            bin_rem[res] = c - weight[i]
            res += 1
        else: # Assign the item to best bin
            bin_rem[bi] -= weight[i]
    return res
 
# Driver code
if __name__ == '__main__':
    weight = [ 2, 5, 4, 7, 1, 3, 8 ]
    weight.sort(reverse=True)
    c = 10
    n = len(weight)
    print("Number of bins required in First Fit : ",
                             firstFit(weight, n, c))