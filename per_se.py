import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('results/performance_per_se_final.csv')

data = data.sort_values(by='rank_at_10_rotate', ascending=True)
