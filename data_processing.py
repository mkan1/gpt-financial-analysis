import json
import csv
import numpy as np

# Load data
with open('dataset/train.json') as f:
    data_dict = json.load(f)

dataset = np.array([
    ('\n'.join(data['pre_text'] 
                + data['post_text'] 
                + [json.dumps(data['table_ori']), json.dumps(data['table'])]) 
     + '\n' + data['qa']['question'], 
     data['qa']['answer'])
    for data in data_dict
])

# Remove questions without an answer
clean_dataset = np.array([(p, c) for (p, c) in dataset if c])

# Save to CSV
with open('formatted_data.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["prompt", "completion"])  # write header
    writer.writerows(clean_dataset)  # write data