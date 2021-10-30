import pandas as pd
import logging

logging.getLogger().setLevel(logging.INFO)

n_tokeep = [50, 100, 150, 200, 250, 300]
min_freq = [500]

# create the dataset in the form pykeen wants it
combo = pd.read_csv('data/bio-decagon-combo.csv', usecols=[0, 1, 2])
combo = combo[['STITCH 1', 'Polypharmacy Side Effect', 'STITCH 2']]
combo.columns = ['head', 'relation', 'tail']

appear = combo['relation'].value_counts().reset_index(name='appearances')

for minimum in min_freq:
    for keep in n_tokeep:
        freq = appear[appear['appearances'] > minimum].nsmallest(n=keep, columns='appearances')
        out = combo.loc[combo['relation'].isin(freq['index'])]
        n_relations = set(out['relation'])
        n_edges = len(out)
        n_drugs = len(set(out['head']).union(set(out['tail'])))
        dupl = out[['head', 'tail']]
        dupl = dupl.groupby(
            dupl.columns.tolist()).size().reset_index().rename(
            columns={0: 'records'}).sort_values(by='records', ascending=False)
        uniq = len(dupl)

        logging.info(f'For minimum = {minimum}, keeping {keep} rarest se \n '
                     f'{n_edges} edges, \n {n_drugs} drugs \n {uniq} unique pairs \n')

        out.to_csv(f'data/rare/rare_{minimum}_{keep}.csv', index=False, sep='\t')


