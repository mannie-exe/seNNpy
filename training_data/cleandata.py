import numpy as np

data = np.genfromtxt('./training_data/export.csv', delimiter=',', dtype=str, encoding='utf-8', usecols=3, max_rows=3)

print(data[0])