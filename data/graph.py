import networkx as nx
from itertools import combinations, groupby
import random
import matplotlib.pyplot as plt
from PIL import Image                                                                                
from IPython.display import display

G = nx.Graph()
n = 20

edges = combinations(range(n), 2)

for i in range(n):
    G.add_node(i, weight = random.randint(1,100))

for _, node_edges in groupby(edges, key=lambda x: x[0]):
    node_edges = list(node_edges)
    random_edge = random.choice(node_edges)
    G.add_edge(*random_edge)
    for e in node_edges:
        if random.random() < 0.5:
            G.add_edge(*e)

labels = {n: G.nodes[n]['weight'] for n in G.nodes}
colors = [G.nodes[n]['weight'] for n in G.nodes]
#print(labels.values())

nx.draw(G, with_labels=True, labels=labels, node_color=colors)
plt.savefig('graph.png', dpi=300, bbox_inches='tight')
plt.show()


