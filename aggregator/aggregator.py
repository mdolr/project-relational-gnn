import csv
import json
import pandas as pd
import sys
import tqdm

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

concepts_file_path = '../data/corrected_concepts_short.csv'
materials_file_path = '../data/materials_short.csv'

output_file_concepts = '../data/corrected_latent_concepts.csv'
output_file_materials = '../data/latent_materials.csv'


# Load the documents
print('Loading documents...')
concepts = pd.read_csv(concepts_file_path)
materials = pd.read_csv(materials_file_path)

concepts['index'] = concepts.index

print(concepts)

# Grab the text from each document
concepts_text = concepts['text']
materials_text = materials['description']

concepts_tags = concepts['index']
materials_tags = materials['tag']

# Split the text word by word
concepts_text = concepts_text.str.split()
materials_text = materials_text.str.split()

concepts_text = concepts_text.to_list()
materials_text = materials_text.to_list()

concepts_text = concepts_text
materials_text = materials_text

# Run Doc2Vec
concepts_documents = [TaggedDocument(doc, [int(concepts_tags[i])])
                      for i, doc in enumerate(concepts_text)]
materials_documents = [TaggedDocument(doc, [int(materials_tags[i])])
                       for i, doc in enumerate(materials_text)]

print('Training concepts model...')
concepts_model = Doc2Vec(
    concepts_documents, vector_size=128, window=2, min_count=3, workers=4)

print('Training materials model...')
materials_model = Doc2Vec(
    materials_documents, vector_size=128, window=2, min_count=3, workers=4)


def get_material_vector(row):
    material_text = materials_text[row['tag']]
    return materials_model.infer_vector(material_text)
    # return concepts_model.infer_vector(text)


def get_concept_vector(row):
    concept_text = concepts_text[row['index']]
    return concepts_model.infer_vector(concept_text)


# For each row in pandas dataframe get the vector
print('Getting concept vectors...')
concepts['vector'] = concepts.apply(get_concept_vector, axis=1)

print('Getting material vectors...')
materials['vector'] = materials.apply(get_material_vector, axis=1)

print('Flattening vectors...')
# Flattenning vector columns
concepts_output = pd.concat([pd.DataFrame(
    concepts[x].values.tolist()) for x in ['vector']], axis=1)
concepts_output.to_csv(output_file_concepts)

# Same for materials
materials_output = pd.concat([pd.DataFrame(
    materials[x].values.tolist()) for x in ['vector']], axis=1)
materials_output.to_csv(output_file_materials)
