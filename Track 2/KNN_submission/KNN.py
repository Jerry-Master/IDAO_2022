import numpy as np
import pandas as pd

filepath_train = './data/dichalcogenides_public/'
filepath_test = './data/dichalcogenides_private/'

targets = pd.read_csv(filepath_train + 'targets.csv', index_col=0)
targets.sort_index(inplace=True)

dist_path = './'
# train_matrix = pd.read_csv(dist_path + 'Train_distance_matrix.csv', index_col=0)
test_matrix = pd.read_csv(dist_path + 'Test_distance_matrix.csv', index_col=0)
print(test_matrix.shape)

predict = []
k = 100
for file in test_matrix.columns:
    dist = test_matrix[file]
    dist[dist==0] = 1000
    dist = dist.sort_values()
    neighs = dist[:k].index
    x = targets.loc[neighs].band_gap
    x = sorted(x)
    for i in range(len(x)-1):
        if x[i+1] - x[i]  < 0.02:
            x[i+1] = x[i]
    x = pd.Series(x)
    predict.append(x.mode().max())
predictions = pd.DataFrame(data={'id': test_matrix.columns, 'predictions': predict})
predictions.to_csv('submission.csv', index=False)