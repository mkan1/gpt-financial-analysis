import json
import numpy as np

f = open('dataset/train.json')
dict = json.load(f)


arr = np.array(
    [
        [
        '\n'.join(data['pre_text'] + data['post_text']),
        json.dumps(data['table_ori']),
        json.dumps(data['table']),
        data['qa']['question'],
        data['qa']['answer']
    ] for data in dict])

X = arr[:, :-1] # input matrix for training data
Y = arr[:, -1]  # output matrix for training data

print(X[0], Y[0])