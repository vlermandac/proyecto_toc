# Model generator for mwvc in scip
# Also writes a file in a MIP model format

from pyscipopt import Model
import numpy as np

# A NetworkX generated graph is assumed
def mwvc(graph, weights, filename):
    model = Model()

    vertices = np.array(graph.nodes)
    edges = np.array(graph.edges)
    x = np.array()
    w = np.array(weights)

    # decision variables
    for v in vertices:
        x[v] = model.addVar(vtype='B')

    # objective function
    model.setObjective(quicksum(w[v]*x[v] for v in vertices), sense='minimize')

    # constraints
    for u,v in edges:
        model.addCons(quicksum(x[u], x[v]) >= 1)

    # writes to output file
    with open(filename, 'w') as lp_file:
        lp_file.write("maximize\nOBJ:" + "".join([f" + {w[node]}x{node+1}" for node in range(graph.number_of_nodes])) + "\n")
        lp_file.write("\nsubject to\n")
        for count, group in enumerate(edges):
            lp_file.write(f"C{count+1}:" + "".join([f" + x{node+1}" for node in sorted(group)]) + " <= 1\n")
        lp_file.write("\nbinary\n" + " ".join([f"x{node+1}" for node in range(graph.number_of_nodes)]) + "\n")


