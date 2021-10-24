import pandas as pd
import numpy as np

combo = pd.read_csv('data/bio-decagon-combo.csv')

# create a dictionary to link polypharmacy side effects to their names
poly_sename_dict = {}
for (se, se_name) in zip(combo['Polypharmacy Side Effect'], combo['Side Effect Name']):
    poly_sename_dict[se] = se_name

combo = combo[['STITCH 1', 'STITCH 2', 'Polypharmacy Side Effect']]
combo.columns = ['node1', 'node2', 'relation']


# Kepp only side effects that occur more than 500 times, as done in original paper. Slow
combo_clean = combo.groupby("relation").filter(lambda x: len(x) > 500)
# export those se
a = combo_clean.relation.unique()
np.savetxt('se.csv', a, delimiter=',', header='poly_side_effects', fmt='%s')

# How many se left
#len(combo_clean.groupby("relation").count())

# Sample 20% for val and test of combo dataset
val_test = combo_clean.groupby('relation').apply(lambda x: x.sample(frac=0.2)).reset_index(drop=True)

# Drop the respective rows from training datatset.
df_all = combo_clean.merge(
    val_test.drop_duplicates(), on=['node1', 'node2', 'relation'], how='left', indicator=True)

c_train = df_all[df_all['_merge'] == 'left_only']
c_train = c_train.drop('_merge', axis=1)

# Sample half of val/test rows to create test dataset
c_test = val_test.groupby('relation').apply(lambda x: x.sample(frac=0.5)).reset_index(drop=True)

# Drop the respective rows from validation dataset
df_dull = val_test.merge(
    c_test.drop_duplicates(), on=['node1', 'node2', 'relation'], how='left', indicator=True)
c_val = df_dull[df_dull['_merge'] == 'left_only']
c_val = c_val.drop('_merge', axis=1)

# Concatenate poly, PPI and target datasets to create training data
ppi = pd.read_csv('data/bio-decagon-ppi.csv')
ppi['relation'] = 'interacts'
ppi.columns = ['node1', 'node2', 'relation']


# We use target, as they do in paper
target = pd.read_csv('data/bio-decagon-targets.csv')
target_all = pd.read_csv('data/bio-decagon-targets-all.csv')

target['relation'] = 'targets'
target.columns = ['node1', 'node2', 'relation']

# Concatenate ppi, targets and combo_training, to create the training dataset
dec_train = ppi.append(target)
dec_train = dec_train.append(c_train)

# Shuffle the datasets
dec_train = dec_train.sample(frac=1)
c_val = c_val.sample(frac=1)
c_test = c_test.sample(frac=1)

# Export to csv
dec_train.to_csv(path_or_buf='raw/decagon_train.csv', header=True, index=False)
c_val.to_csv(path_or_buf='raw/decagon_validation.csv', header=True, index=False)
c_test.to_csv(path_or_buf='raw/decagon_test.csv', header=True, index=False)

# Create mini files with ~40.000 rows, ~5.000 rows
dec_sample = dec_train.sample(frac=0.01)
c_val_sample = c_val.sample(frac=0.01)
c_test_sample = c_test.sample(frac=0.01)
dec_sample.to_csv(path_or_buf='raw/sample.csv', header=True, index=False)
c_val_sample.to_csv(path_or_buf='raw/val_sample.csv', header=True, index=False)
c_test_sample.to_csv(path_or_buf='raw/test_sample.csv', header=True, index=False)
