import numpy as np
import scipy as sp
import pandas as pd
import itertools
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("Pv")
parser.add_argument("Py")
parser.add_argument("P1")
parser.add_argument("P2")
parser.add_argument("P3")
parser.add_argument("MAX_INT")
parser.add_argument("--out_file")

args = parser.parse_args()
o_f = "Qmat.txt"
if args.out_file:
    o_f = args.out_file


LvFT = 0
LyFT = 0
L2FT = 0
FREE_TERM = 0

MAX_INT = args.MAX_INT
nodes_frame = pd.read_csv("/workspace/task-2-nodes.csv", header=None, index_col=0).sort_values(1)
edges_frame = pd.read_csv("/workspace/task-2-adjacency_matrix.csv", index_col=0, na_values=["-"])

# Sorting nodes such that initial is 0, and all crossings are at the beginning
initial_node = "Вокзал"
node_names = nodes_frame.index.tolist()
node_names.remove(initial_node)
node_names.insert(0, initial_node)
assert len(node_names) == 57 and node_names[0]==initial_node, "Sonething goes wrong with reordering initial node"
print(node_names)

A = edges_frame.reindex(index=node_names, columns=node_names).to_numpy(na_value=MAX_INT, copy=True).astype(int)
B = nodes_frame.reindex(index=node_names).to_numpy(na_value=MAX_INT, copy=True).astype(int).reshape((-1))
sights_start = np.searchsorted(B, 0, side='right')  # Start of nodes which should be visited only once

NUM_BUSES = 15
MAX_PEOPLE = 10
MAX_TACTS = 15
NUM_TACTS = MAX_TACTS - 1
NUM_NODES = len(node_names)
NUM_PEOPLE = MAX_PEOPLE + 1

# Array will be encoded as {v[b,i,j], y[b,k]}
# b = 0,...,NUM_BUSES-1
# i = 1,...,NUM_TACTS-1
# j = 0,...,NUM_NODES-1
# k = 0,...,MAX_PEOPLE

def get_index(inp : list) -> int:
    """
    Returns integer index in big array of element v,b,i,j or y,b,k
    """
    var = inp
    if var[0] == "v":
        return (MAX_TACTS-1)*NUM_NODES*int(var[1]) + NUM_NODES*(int(var[2]) - 1) + int(var[3])
    elif var[0] == "y":
        return NUM_BUSES*NUM_TACTS*NUM_NODES + (MAX_PEOPLE + 1)*int(var[1]) + int(var[2])

# H
print("Started H")
SHAPE = NUM_BUSES*NUM_TACTS*NUM_NODES + NUM_BUSES*NUM_PEOPLE
H = np.zeros((SHAPE, SHAPE))
for b, j in itertools.product(range(NUM_BUSES), range(NUM_NODES)):
    ind = get_index(["v",b,1,j])
    H[ind, ind] += A[0,j]
    ind = get_index(["v",b,NUM_TACTS,j])
    H[ind, ind] += A[j,0]

for b, i, j, jp in itertools.product(range(NUM_BUSES), range(1,MAX_TACTS-1), range(NUM_NODES), range(NUM_NODES)):
    H[get_index(["v",b,i,j]), get_index(["v",b,i+1,jp])] += A[j,jp]

# Lv
print("Started Lv")
Lv = np.zeros((SHAPE, SHAPE))
for b, i, j in itertools.product(range(NUM_BUSES), range(1,MAX_TACTS), range(NUM_NODES)):
    Lv[get_index(["v",b,i,j]), get_index(["v",b,i,j])] -= 2
for b, i, j, jp in itertools.product(range(NUM_BUSES), range(1,MAX_TACTS), range(NUM_NODES), range(NUM_NODES)):
    Lv[get_index(["v",b,i,j]), get_index(["v",b,i,jp])] += 1

# Ly
print("Started Ly")
Ly = np.zeros((SHAPE, SHAPE))
for b, k in itertools.product(range(NUM_BUSES), range(NUM_PEOPLE)):
    ind = get_index(["y", b, k])
    Ly[ind, ind] -= 2
    
for b, k, kp in itertools.product(range(NUM_BUSES), range(NUM_PEOPLE), range(NUM_PEOPLE)):
    Ly[get_index(["y", b, k]), get_index(["y", b, kp])] += 1

# L1
print("Started L1")
L1 = np.zeros((SHAPE, SHAPE))
for b in range(NUM_BUSES):
    for i,ip,j,jp in itertools.product(range(1,MAX_TACTS), range(1,MAX_TACTS), 
                                    range(sights_start, NUM_NODES), range(sights_start, NUM_NODES)):
        L1[get_index(["v", b, i, j]), get_index(["v", b, ip, jp])] += B[j]*B[jp]
    
    for k,kp in itertools.product(range(1, NUM_PEOPLE), range(1, NUM_PEOPLE)):
        L1[get_index(["y", b, k]), get_index(["y", b, kp])] += k*kp
        
    for i,j,k in itertools.product(range(1, MAX_TACTS), range(sights_start, NUM_NODES), range(1, NUM_PEOPLE)):
        L1[get_index(["v", b, i, j]), get_index(["y", b, k])] -= 2 * B[j] * k

# L2
print("Started L2")
L2 = np.zeros((SHAPE, SHAPE))
for j in range(sights_start, NUM_NODES):
    for b,i in itertools.product(range(NUM_BUSES), range(1, MAX_TACTS)):
        ind = get_index(["v", b, i, j])
        L2[ind, ind] -= 2
        for bp,ip in itertools.product(range(NUM_BUSES), range(1, MAX_TACTS)):
            ind2 = get_index(["v", bp, ip, j])
            L2[ind, ind2] += 1
        
# L3
print("Started L3")
L3 = np.zeros((SHAPE, SHAPE))
for i,j in itertools.product(range(1, MAX_TACTS), range(1, NUM_NODES)):
    for b,bp in itertools.combinations(range(NUM_BUSES), 2):
        L3[get_index(["v", b, i, j]), get_index(["v", bp, i, j])] += 1

NUM_SIGHTS = NUM_NODES - sights_start
Lv_ft = NUM_BUSES * NUM_TACTS
Ly_ft = NUM_BUSES
L2_ft = NUM_SIGHTS

# with open("/workspace/weights.json") as f:
#     weights = json.load(f)
#     Pv = weights["Pv"]
#     Py = weights["Py"]
#     P1 = weights["P1"]
#     P2 = weights["P2"]
#     P3 = weights["P3"]
#

Pv = args.Pv
Py = args.Py
P1 = args.P1
P2 = args.P2
P3 = args.P3

 
LvFT = Lv_ft*Pv
LyFT = Ly_ft*Py
L2FT = L2_ft*P2
FREE_TERM = LvFT + LyFT + L2FT


# Combining
print("Combining")

Qnp = H + Pv*Lv + Py*Ly + P1*L1 + P2*L2 + P3*L3
Qnp = np.triu(Qnp + np.triu(Qnp.T)) / 2

Q = sp.sparse.coo_array(Qnp)

print("Start writing file")
with open("/workspace/"+o_f, mode='wt') as out_f:
    out_f.write(f"{SHAPE} {len(Q.row)}\n")
    for c,r,d in zip(Q.col, Q.row, Q.data):
        out_f.write(f"{r+1} {c+1} {d}\n")



def check_constraints(Lmat, vec, free_term=0):
    return vec.T @ Lmat @ vec + free_term


def verify_constraints():
    with open("/workspace/output.json", mode='rt') as in_f:
        res = json.load(in_f)
    sol_vec = np.asarray(res["Solution"], dtype=int)
    print("Objective is " + str(res["Objective"] + FREE_TERM))
    print(f"Constraint Lv={check_constraints(Lv, sol_vec, Lv_ft)}")
    print(f"Constraint Ly={check_constraints(Ly, sol_vec, Ly_ft)}")
    print(f"Constraint L1={check_constraints(L1, sol_vec, 0)}")
    print(f"Constraint L2={check_constraints(L2, sol_vec, L2_ft)}")
    print(f"Constraint L3={check_constraints(L3, sol_vec, 0)}")
    return



verify_constraints()