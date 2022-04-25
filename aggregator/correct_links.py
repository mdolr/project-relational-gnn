from enum import unique
import pandas as pd

# An util function to correct the concept_tag column
# which is bugged in the parser
# without having to re-parse the whole database

# Without this function we end up with links between materials and concepts that are not linked
# because the parser increments the concept_tag everytime instead of
# using an existing concept_tag when the concept is already known

links_file_path = '../data/links_short.csv'
concepts_file_path = '../data/concepts_short.csv'

links = pd.read_csv(links_file_path, index_col=0)
concepts = pd.read_csv(concepts_file_path, index_col=0)

seen_concepts = []


def correct_concept_tag(row):
    return concepts.loc[concepts['slug'] == row['concept_slug']].index[0]


def incremental_concept_tag(row):
    if row['concept_slug'] in seen_concepts:
        return seen_concepts.index(row['concept_slug'])
    else:
        seen_concepts.append(row['concept_slug'])
        return len(seen_concepts) - 1


links['concept_tag'] = links.apply(correct_concept_tag, axis=1)
links['incremental_concept_tag'] = links.apply(incremental_concept_tag, axis=1)
output_file_path = '../data/corrected_links_short.csv'

links.to_csv(output_file_path)
print(links)

unique_concepts_tags = list(links['concept_tag'].unique())
concepts = concepts.iloc[unique_concepts_tags]
concepts.to_csv('../data/corrected_concepts_short.csv')
