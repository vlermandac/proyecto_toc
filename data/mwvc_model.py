# Model generator for mwvc in scip
# Also writes a file in a MIP model format

from pyscipopt import Model
import numpy as np
from pyscipopt.scip import Expr, ExprCons, Term, quicksum
import networkx as nx
import ecole
import random
from itertools import combinations, groupby

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
        lp_file.write("maximize\nOBJ:" + "".join([f" + {w[node]}x{node+1}" for node in range(graph.number_of_nodes)]) + "\n")
        lp_file.write("\nsubject to\n")
        for count, group in enumerate(edges):
            lp_file.write(f"C{count+1}:" + "".join([f" + x{node+1}" for node in sorted(group)]) + " <= 1\n")
        lp_file.write("\nbinary\n" + " ".join([f"x{node+1}" for node in range(graph.number_of_nodes)]) + "\n")

class mwvc_scip:
    def __init__(self, nodes):
        self.nodes = nodes

    def generate_graph(self):
        G = nx.Graph()

        edges = combinations(range(self.nodes), 2)

        for i in range(self.nodes):
            G.add_node(i, weight = random.randint(1,100))

        for _, node_edges in groupby(edges, key=lambda x: x[0]):
            node_edges = list(node_edges)
            random_edge = random.choice(node_edges)
            G.add_edge(*random_edge)
            for e in node_edges:
                if random.random() < 0.5:
                    G.add_edge(*e)
        return G


    def generate_model(self):
            model = Model()
            graph = self.generate_graph()
            vertices = np.array(graph.nodes)
            edges = np.array(graph.edges)

            x = np.array()

            labels =  {n: graph.nodes[n]['weight'] for n in graph.nodes}
            w = np.array(labels.values())

            # decision variables
            for v in vertices:
                x[v] = model.addVar(vtype='B')

            # objective function
            model.setObjective(quicksum(w[v]*x[v] for v in vertices), sense='minimize')

            # constraints
            for u,v in edges:
                model.addCons(quicksum(x[u], x[v]) >= 1)
            
            return model
        
        


