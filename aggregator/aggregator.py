import csv
import json
import pandas as pd
import sys
import tqdm

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

links_file_path = '../data2/links.csv'
concepts_file_path = '../data2/concepts.csv'
materials_file_path = '../data2/materials.csv'
"""
concepts_documents = [TaggedDocument(doc, [i])
                      for i, doc in enumerate(common_texts)]
resources_documents = [TaggedDocument(doc, [i])
                       for i, doc in enumerate(common_texts)]
"""

# Load the documents
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

final_dimension = 128 + 128 + 1

print('Saving vectors...')
with open(links_file_path, newline='') as links:
    with open('../data2/aggregated_vectors.csv', 'w', newline='') as aggregated_vectors:

        reader = csv.reader(links, delimiter=',', quotechar='|')
        for i, row in enumerate(reader):
            if i != 0:
                material_tag = int(row[0])
                concept_tag = int(row[1])

                material_text = materials_text[material_tag]
                concept_text = concepts_text[concept_tag]

                material_vector = materials_model.infer_vector(material_text)
                concept_vector = concepts_model.infer_vector(concept_text)

                final_vector = list(material_vector) + \
                    list(concept_vector) + [row[4]]

                # Transform all elements of final_vector to string
                final_vector = [str(x) for x in final_vector]

                aggregated_vectors.write(','.join(final_vector) + '\n')
