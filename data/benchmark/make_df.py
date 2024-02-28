import pandas as pd
import numpy as np

# read all xlsx files in the folder
list_intensities = []
for batch in ["Batch1_0108 DATA.xlsx", "Batch2_0110 DATA.xlsx", "Batch3_0124 DATA.xlsx", "Batch4_0219 DATA.xlsx", "Batch5_0221 DATA.xlsx", "Batch6_0304 DATA.xlsx", "Batch7_0306 DATA.xlsx"]:
    intensities = pd.read_excel(f"data/benchmark/{batch}", sheet_name="intensities").iloc[:, 2:].transpose()
    # injections = pd.read_excel(f"data/benchmark/{batch}", sheet_name="injections")
    inds = intensities.index
    # split all indices on " / " and take the second element
    inds = [ind.split(" / ")[1] for ind in inds]
    # remove burnin and water from df intensities

    inds = [i for i, ind in enumerate(inds) if ind not in ["burnin", "water"]]
    intensities = intensities.iloc[inds, :]
    # split on _ and take the second element
    inds = [ind.split("_")[1] for ind in intensities.index]

    # Add a batch column
    # Make that the second column
    intensities["batch"] = batch
    cols = intensities.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    intensities = intensities[cols]
    
    intensities["label"] = inds
    cols = intensities.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    intensities = intensities[cols]
    
    intensities["names"] = intensities.index
    cols = intensities.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    intensities = intensities[cols]
    
    list_intensities.append(intensities)
    if len(intensities) % 3 != 0:
        print(batch, len(intensities))

# merge all dataframes
df = pd.concat(list_intensities, ignore_index=False)

# Verify that all names are in triplicate
names = df["names"].to_numpy()
names = [name.split(" / ")[1] for name in names]
# assert sum([names.count(name) == 3 for name in names]) == len(names)
# find which name is not in triplicate
for name in names:
    if names.count(name) != 3:
        print(name, names.count(name))

# Get the number of batch each name is in
batches = df["batch"].to_numpy()
batches = [batch.split("_")[0] for batch in batches]
batches = np.array([int(batch.split("Batch")[1]) for batch in batches])

# Get the indices where column name is all the same
unique_names = np.unique(names)
n_batches = {}
for name in unique_names:
    n_batches[name] = len(np.unique(batches[np.argwhere(np.array(names)==name).flatten()]))
min_batch = min(n_batches.values())
max_batch = max(n_batches.values())
df.iloc[np.argwhere(np.array(names)==names[0]).flatten(), :]

# Plot the number of batches each name is in
# Center the bars on the x-axis
import matplotlib.pyplot as plt
from collections import Counter
c = Counter(n_batches.values())
plt.bar(c.keys(), c.values())
plt.xlabel("Number of batches")
plt.ylabel("Frequency")
plt.title("Number of batches each sample is in")
plt.savefig("data/benchmark/n_batches.png")
plt.close()

# Number of samples per batch
c = Counter(batches)
plt.bar(c.keys(), c.values())
plt.xlabel("Batch")
plt.ylabel("Number of samples")
plt.title("Number of samples per batch")
plt.savefig("data/benchmark/n_samples_per_batch.png")
plt.close()

# Number of samples per label
labels = df["label"].to_numpy()
c = Counter(labels)
plt.bar(c.keys(), c.values())
plt.xlabel("Label")
plt.ylabel("Number of samples")
plt.title("Number of samples per label")
plt.savefig("data/benchmark/n_samples_per_label.png")
plt.close()

# Batch with the least number of samples
min_batch = min(c.values())
min_batch = [batch for batch, n in c.items() if n == min_batch]
min_batch = df[df["batch"].str.contains(min_batch[0])]


# save the dataframe
df.to_csv("data/benchmark/intensities.csv", index=False)
