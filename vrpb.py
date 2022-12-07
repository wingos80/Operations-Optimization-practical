import matplotlib.pyplot as plt
import gurobipy as grb
import numpy as np
import os
import pandas as pd
import argparse
import timeit


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

def model(dataframe, limit=20):
    model = grb.Model(name='VRPB')

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
    line,back,depo = 0,0,0
    circle_size = 75

    # looping through all nodes and populating all variable, set, and parameters dictionaries
    for node1 in nodes:
        if node1 >= limit:
            break # stopping the code from making too many nodes, otherwise it will take too long to run
        node1_type = types[node1]
        node1_coords = np.array([x_coords[node1], y_coords[node1]])
        V[node1] = node1

        if node1_type == 'L':
            L[node1] = node1
            L0[node1] = node1
            Cu[node1] = node1
            D[node1] = demands[node1]

            # plotting the linehaul node
            if line == 0:
                plt.scatter(node1_coords[0], node1_coords[1], color='C1', label='Linehaul', zorder=3, s=circle_size)
                line=1
            elif line == 1:
                plt.scatter(node1_coords[0], node1_coords[1], color='C1', zorder=3, s=circle_size)
            plt.annotate(node1, (node1_coords[0] , node1_coords[1]+0.2), fontsize=8, zorder=4, color='black')
            # plt.scatter(node1_coords[0], node1_coords[1], color='C1', zorder=3)


        elif node1_type == 'B':
            B[node1] = node1
            B0[node1] = node1
            Cu[node1] = node1
            D[node1] = demands[node1]

            # plotting the backhaul node
            if back == 0:
                plt.scatter(node1_coords[0], node1_coords[1], color='C2', label='Backhaul', zorder=3, s=circle_size)
                back=1
            elif back == 1:
                plt.scatter(node1_coords[0], node1_coords[1], color='C2', zorder=3, s=circle_size)
            plt.annotate(node1, (node1_coords[0] , node1_coords[1]+0.2), fontsize=8, zorder=4, color='black')
            # plt.scatter(node1_coords[0], node1_coords[1], color='C2', zorder=3)

        elif node1_type == 'D':
            L0[node1] = node1
            B0[node1] = node1

            # plotting the depot node
            if depo == 0:
                plt.scatter(node1_coords[0], node1_coords[1], color='C0', label='Depot', zorder=3, s=circle_size)
                depo=1
            elif depo == 1:   
                plt.scatter(node1_coords[0], node1_coords[1], color='C0', zorder=3, s=circle_size)
            plt.annotate(node1, (node1_coords[0] , node1_coords[1]+0.2), fontsize=8, zorder=4, color='black')
            # plt.scatter(node1_coords[0], node1_coords[1], color='C0', zorder=3)
        
        for node2 in nodes:
            if node2 >= limit:
                break
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

    Q = 60

    # solving the knapscak problem to calculate minimum Kl and Kb
    Kl, Kb = Get_K(L, B, D, Q)
    print(f'kl and kb are: {Kl} {Kb}')
    # Kl, Kb = 3, 3

    model.update()

    ##############################
    ### Create the constraints ###
    ##############################
    model.addConstr(grb.quicksum(s[i,j] for i in L0 for j in L) == len(L), name='c1') # C1  

    for j in L:
        model.addConstr(grb.quicksum(l[i,j] for i in L0 if i!=j) - grb.quicksum(l[j,k] for k in L0 if k !=j) == D[j], name=f'c2_{j}') #C2   
        
        model.addConstr(grb.quicksum(s[i,j] for i in L0) == 1, name=f'c3_{j}') # C3
        
        model.addConstr(grb.quicksum(s[j,k] for k in L) + grb.quicksum(xi[j,k] for k in B0) == grb.quicksum(s[i,j] for i in L0), name=f'c4_{j}') #C4
        
        # BUG, bad constraint
        # model.addConstr(l[0,j] <= Q*s[0,j], name=f'c5_{0}{j}') # C5 bad
        for i in L0:
            model.addConstr(l[i,j] <= Q*s[i,j], name=f'c5_{i}{j}') # C5 good 

    # model.addConstr(grb.quicksum(s[0,j] for j in L) == Kl, name='c6') # C6.1
    model.addConstr(grb.quicksum(s[0,j] for j in L) >= Kl, name='c6') # C6.2

    model.addConstr(grb.quicksum(s[i,j] for i in B for j in B0) == len(B), name='c7') # C7

    for j in B:
        model.addConstr(grb.quicksum(l[i,j] for i in B if i!=j) - grb.quicksum(l[j,k] for k in B0 if k !=j) == -D[j], name=f'c8_{j}') #C8 

        # # # BUG in C9 and C10, order of i and j is wrong in the paper, show daniel why theres bug here
        # model.addConstr(grb.quicksum(s[i,j] for i in B0) == 1, name=f'c9_{j}') # C9

        # model.addConstr(grb.quicksum(s[k,j] for k in B) + grb.quicksum(xi[k,j] for k in L) == grb.quicksum(s[i,j] for i in B0), name=f'c10_{j}') #C10 
        
        model.addConstr(grb.quicksum(s[j,i] for i in B0) == 1, name=f'c9_{j}') # C9

        model.addConstr(grb.quicksum(s[k,j] for k in B) + grb.quicksum(xi[k,j] for k in L) == grb.quicksum(s[j,i] for i in B0), name=f'c10_{j}') #C10 

        for i in B0:
            model.addConstr(l[j,i] <= Q*s[j,i], name=f'c11_{j}{i}') # C11

    model.addConstr(Kb <= grb.quicksum(s[i, 0] for i in B), name='c12.1') # C12.1
    model.addConstr(grb.quicksum(s[i, 0] for i in B) <= grb.quicksum(s[0,j] for j in L), name='c12.2') # C12.2
            
    model.addConstr(grb.quicksum(s[i,0] for i in B) + grb.quicksum(xi[i,0] for i in L) == grb.quicksum(s[0,j] for j in L), name='c13') # C13

    for i in V:
        for j in V:
            model.addConstr(s[i,j]+s[j,i] <= 1, name=f'c14_{i}_{j}') # C14

    model.addConstr(grb.quicksum(s[i,j] for i in B for j in L) == 0, name='c15')     # C15
    model.addConstr(grb.quicksum(s[0,j] for j in B) == 0, name='c16')                # C16
    model.addConstr(grb.quicksum(s[i,j] for i in L for j in B0) == 0, name='c17')    # C17
    # model.addConstr(grb.quicksum(s[0,j] for j in V) == 3, name='c18') # constraint which will set the number of trucks to 3

    model.update()

    #####################
    ### Set objective ###
    #####################


    obj = grb.quicksum(C[i,j]*s[i,j] for i in V for j in V) + grb.quicksum(C[i,j]*xi[i,j] for i in L for j in B0)    # adding the cost of arcs
    # obj += 9999*grb.quicksum(s[0,i] for i in V)    # adding number of trucks used as cost
    model.setObjective(obj, grb.GRB.MINIMIZE)

    model.update()

    ########################
    ## Optimize the model ##
    ########################

    ### Run and optimize model

    toc = timeit.default_timer()

    model.optimize()
    model.write(f'test_{limit}.lp')

    tic = timeit.default_timer()

    print(f'Objective function value: {model.objVal}')

    for v in model.getVars():
        if v.x != 0:
            print(f'{v.varName} = {v.x}')

            # drawing the paths that are used
            if v.varName[0] == 's' or v.varName[0] == 'x' :
                node1 = int(v.varName.split("_")[1])
                node2 = int(v.varName.split("_")[2])
                node1_coords = (x_coords[node1], y_coords[node1])
                node2_coords = (x_coords[node2], y_coords[node2])
                mid_coords = ((node1_coords[0] + node2_coords[0])/2, (node1_coords[1] + node2_coords[1])/2)
                if types[node2] == 'L':
                    color = 'C1'
                else:
                    color = 'C2'
                plt.plot([node1_coords[0], node2_coords[0]], [node1_coords[1], node2_coords[1]], color=color, zorder=2, alpha=0.8)
                plt.arrow(mid_coords[0], mid_coords[1], (node2_coords[0] - node1_coords[0])/10, (node2_coords[1] - node1_coords[1])/10, head_width=0.7, head_length=0.7, color=color, alpha=0.8)

    print(f'\n\n******Time taken to optimize: {round(tic-toc,6)} seconds******')


    plt.legend()
    plt.tight_layout()
    plt.axis('off')
    # plt.grid()
    plt.show()

    plt.cla()

if __name__ == '__main__':
    
    # Adding wildcards for calling the script
    parser = argparse.ArgumentParser(description='Runs the BAD vrpb algorithm')
    parser.add_argument('--excel', type=str, default=None,
                        help='The excel file that will be used')
    parser.add_argument('--sheet', type=str, default=None,
                        help='The sheet that will be used from the excel file')
    args = parser.parse_args()
    print(args.excel)
    
    ###################
    ### Import data ###
    ###################
    
    # Get all instances
    cwd = os.getcwd()
    full_list = os.listdir(cwd)

    excel_file = args.excel + ".xlsx"
    print('\n\nrunning for excel:',excel_file,'\n\n')

    sheet_name = args.sheet if args.sheet else 0

    # Load data for this instance
    dataframe = pd.read_excel(os.path.join(cwd,excel_file),sheet_name=sheet_name) # edges are basically the variables in a VRP for ONE TRUCK!!

    for i in range(4):
        model(dataframe, 20+2*i)
