import numpy as np
import pandas as pd


# Import xlsx file
df = pd.read_excel('data/PXD015912/Mapping_file_PXD015912.xlsx', sheet_name='PXD015912')
matrix = pd.read_csv('data/PXD015912/Peptide_intensity_matrix_b9369842-f3cf-4383-9e5c-6734dadcfbc9.csv', sep='\t')

matrix.index = matrix.loc[:, 'Peptide'].to_numpy()
# 18113 rows × 1556 columns

# The last 3 columns are protein, peptide and mod peptide names
matrix = matrix.iloc[:, :-3]
matrix = matrix.transpose()

inds = np.array(matrix.index)
filenames = np.array([str(np.char.lower(x)) for x in df['Filename'].to_numpy()])

new_inds = [np.argwhere(x==inds).flatten()[0] for x in filenames if np.char.lower(x) in inds]
new_inds = [np.argwhere(ind==filenames).flatten()[0] for ind in inds]

new_df = df.iloc[new_inds, :]

infos = np.array([str(np.char.lower(x)) for x in new_df['Information'].to_numpy()])
days = np.array([x.split("_")[0] for x in infos])
batches = np.array([x.split("_")[1] for x in infos])
labels = np.array([' '.join(x.split(" ")[2:]) for x in infos])

assert len(days) == len(batches) == len(new_df)
assert sum(['day' in x in x for x in days]) == sum(['m' in x in x for x in batches]) == len(days)

days = pd.DataFrame(days.reshape([-1, 1]), index=matrix.index, columns=['day'])
batches = pd.DataFrame(batches.reshape([-1, 1]), index=matrix.index, columns=['batches'])
labels = pd.DataFrame(labels.reshape([-1, 1]), index=matrix.index, columns=['labels'])

new_matrix = pd.concat([labels, days, batches, matrix], axis=1)

db = ['_'.join([x, y]) for x, y in zip(days.to_numpy().flatten(), batches.to_numpy().flatten())]

# Save new matrix
new_matrix.to_csv('data/PXD015912/matrix.csv')
