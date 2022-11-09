import pandas as pd
from matplotlib import pyplot as plt
from collections import defaultdict
import numpy as np
import json
from sklearn.decomposition import PCA

mono = pd.read_csv('data/bio-decagon-mono.csv', header=0, names=['drug', 'se', 'se_name'])

se_names = mono.drop_duplicates(subset=['se', 'se_name']).drop(['drug'], axis=1)
freq = mono['se'].value_counts().reset_index(
    name='appearances').rename(columns={'index': 'se'})
mono = mono.merge(freq, on='se', how='left')


# Some interesting exports regarding the dataset

freq_names = freq.merge(se_names, on='se', how='left')
freq_names.to_csv('results/frequent_mono_se.csv')


# For a start let's see the frequency of the se
# ~5000 se appear in 5 or less drugs
# ~300 in more than 100 ~30 in more than 200

plt.hist(freq.appearances, bins=60, color='r')
plt.xlabel('number of appearances')
plt.ylabel('number of side effects')
plt.xlim(0, 150)
plt.title('distribution of side effects frequency')
plt.show()

drugs = mono.groupby('drug')['se'].apply(list).reset_index(name='side_effects')
n_of_se = [len(se) for se in drugs['side_effects']]
n_of_se.sort(reverse=True)

plt.hist(n_of_se, bins=100, color='r')
plt.xlabel('number of side effects')
plt.ylabel('number of drugs')
plt.title('distribution of the number of side effects per drug')
plt.show()

# create jsons with feature vectors
mono_rem_rare = mono[mono['appearances'] > 5]
mono_rem_rare_over200 = mono_rem_rare[mono_rem_rare['appearances'] < 200]
mono_rem_rare_over100 = mono_rem_rare[mono_rem_rare['appearances'] < 100]
mono_mini = mono_rem_rare[mono_rem_rare['appearances'] > 250]

datasets = {'mono': mono,
            'mono_rem_rare': mono_rem_rare,
            'mono_rem_rare_over200': mono_rem_rare_over200,
            'mono_rem_rare_over100': mono_rem_rare_over100,
            'mono_mini': mono_mini}

for name, dataset in datasets.items():
    # create a dictionary to link drugs to their mono se
    drug_se_dict = defaultdict(set)
    for (drug, se) in zip(dataset['drug'], dataset['se']):
        drug_se_dict[drug].add(se)

    side_effects = dataset['se'].unique()
    side_effects.sort()
    side_effects = list(side_effects)

    # create a dict of lists holding the feature vectors forevery drug (which has mono se)
    drug_features = {}
    for drug in drug_se_dict:
        vector = np.zeros(len(side_effects))
        mono_se_found_indexes = [side_effects.index(mono_se) for mono_se in
                                 drug_se_dict[drug]]
        vector[mono_se_found_indexes] = 1
        vector = list(vector)
        drug_features[drug] = vector

    # drug features need to be list for that
    with open(f'results/feature_vectors_{name}.json', 'w') as f:
        json.dump(drug_features, f, indent=2)


    # PCA
    pca = PCA(n_components=100)
    drug_features_pca = pca.fit_transform(drug_features)
    print(drug_features_pca.explained_variance_ratio_)



    foo = pd.DataFrame(drug_features)
    cols = foo.columns
    # inverse document frequency
    foo.loc[:, 'total'] = foo.sum(axis=1)
    foo.loc[:, 'percent'] = foo.shape[1] / foo['total']
    foo.loc[:, 'idf'] = np.log10(foo['percent'])
    # term frequency
    foo[cols] = foo[cols] / foo[cols].sum()
    # tf * idf
    foo[cols] = foo[cols].multiply(foo['idf'], axis=0).round(decimals=5)
    foo = foo[cols]

    # back to same json format
    boo = foo.to_dict('list')
    with open(f'results/tfidf_vectors_{name}.json', 'w') as f:
        json.dump(boo, f, indent=2)



