import torch
import pandas as pd
from torch_geometric.data import HeteroData


concepts_path = '../data2/latent_concepts.csv'
materials_path = '../data2/latent_materials.csv'
links_path = '../data2/links_short.csv'


def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = df.values

    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = df['target']
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


materials, materials_mapping = load_node_csv(
    materials_path, index_col=0)


concepts, concepts_mapping = load_node_csv(
    concepts_path, index_col=0)

print(materials)

# Initialize a HeteroData object
data = HeteroData(
    materials={'x': materials}, concepts={'x': concepts})


edge_index, edge_label = load_edge_csv(
    links_path,
    src_index_col='material_tag',
    src_mapping=materials_mapping,
    dst_index_col='concept_tag',
    dst_mapping=concepts_mapping,
)

data['materials', 'links', 'concepts'].edge_index = edge_index
data['materials', 'links', 'concepts'].edge_label = edge_label

print(data)
