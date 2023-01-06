from pyscipopt import Model
import numpy as np
from pyscipopt.scip import quicksum
import networkx as nx
import ecole
import random
from itertools import combinations, groupby


class mwvc_scip:
    def __init__(self, nodes=2):
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

            x = [0] * len(vertices)
            

            labels =  {n: graph.nodes[n]['weight'] for n in graph.nodes}
            w =  list(labels.values())
            

            # decision variables
            for v in vertices:
                x[v] = model.addVar(vtype='B')

            # objective function
           
            model.setObjective(quicksum(w[v]*x[v] for v in vertices), sense='minimize')

            # constraints
            for u,v in edges:
                model.addCons(x[u]+ x[v] >= 1)
            model = ecole.scip.Model.from_pyscipopt(model)
            return model

if __name__ == "__main__":
        
    instancia = mwvc_scip(10)

    print(instancia.generate_model())
   
   