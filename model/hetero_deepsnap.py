import torch
import numpy as np
import pandas as pd
from deepsnap.hetero_graph import HeteroGraph
import pickle
import networkx as nx

concepts_path = '../data2/latent_concepts.csv'
materials_path = '../data2/latent_materials.csv'
links_path = '../data2/links_short.csv'

concepts = pd.read_csv(concepts_path, index_col=0)
materials = pd.read_csv(materials_path, index_col=0)
links = pd.read_csv(links_path, index_col=0)

graph = nx.Graph()

print('Adding concepts...')
for index, row in concepts.iterrows():
    graph.add_node(index, node_type='concept',
                   node_feature=torch.tensor(np.array(row)))

print('Adding materials...')
for index, row in materials.iterrows():
    graph.add_node(index, node_type='material',
                   node_feature=torch.tensor(np.array(row)))

print('Adding links...')
for index, row in links.iterrows():
    graph.add_edge(row['material_tag'], row['concept_tag'],
                   edge_type='link', edge_label=row['target'])


data = HeteroGraph(graph)
pickle.dump(data, open('../data2/data3.pkl', 'wb'))
