# Model generator for mwvc in scip

from pyscipopt import Model
import numpy as np

def mwvc(graph, weights):
    model = Model()

    vertices = graph.vertices
    edges = graph.edges
    x = np.array()
    w = np.array()

    # decision variables
    for v in vertices:
        x[v] = model.addVar(vtype='B')

    # objective function
    model.setObjective(quicksum(w[v]*x[v] for v in vertices), sense='minimize')

    # constraints
    for u,v in edges:
      model.addCons(quicksum(x[u], x[v]) >= 1)


