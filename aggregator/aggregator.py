import csv
import json
import pandas as pd
import sys
import tqdm

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

links_file_path = '../data2/links_short.csv'
concepts_file_path = '../data2/concepts_short.csv'
materials_file_path = '../data2/materials_short.csv'

output_file_concepts = '../data2/latent_concepts.csv'
output_file_materials = '../data2/latent_materials.csv'


# Load the documents
print('Loading documents...')
concepts = pd.read_csv(concepts_file_path)
materials = pd.read_csv(materials_file_path)

# Grab the text from each document
concepts_text = concepts['text']
materials_text = materials['description']

concepts_tags = concepts['tag']
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
    concept_text = concepts_text[row['tag']]
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

"""
with open(materials_file_path, newline='') as materials_file:
    with open(output_file_materials, 'w', newline='') as output_materials:
        for i, row in enumerate(materials_file):
            if i != 0:
                tag = int(row[0])

                material_text = materials_text[tag]
                material_vector = materials_model.infer_vector(material_text)

                final_vector = [tag] + list(material_vector)
                final_vector = [str(x) for x in final_vector]

                output_materials.write(','.join(final_vector) + '\n')


print('Writing concepts...')
with open(concepts_file_path, newline='') as concepts_file:
    with open(output_file_concepts, 'w', newline='') as output_concepts:
        for i, row in enumerate(concepts_file):
            if i != 0:
                tag = int(row[0])
                
                concept_text = concepts_text[tag]
                concept_vector = concepts_model.infer_vector(concept_text)

                final_vector = [tag] + list(concept_vector)
                final_vector = [str(x) for x in final_vector]

                output_concepts.write(','.join(final_vector) + '\n')

"""
