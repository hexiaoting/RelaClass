import os
import json
from tqdm import tqdm

def process(dataset_name, specific_file, data_dir=None):
    if data_dir is None:
        data_dir = os.path.join("/home/hewenting/data_preprocess/", dataset_name)
    
    with open(specific_file) as f:
        new_lines = f.readlines()
    
    results = []
    replaced_num = 0
    original_file = os.path.join(data_dir, f'{dataset_name}_train.json')
    with open(original_file) as f:
        readin = f.readlines()
        for line in tqdm(readin):
            item = eval(line)
            if item['label'][0] == 23 and item['label'][1] == 26:
                line = new_lines[replaced_num]
                replaced_num += 1
            results.append(line.strip('\n'))

    output_file = os.path.join(data_dir, f'Amazon-531-enhanced_train.json')

    with open(output_file, 'w') as f:
        for item in results:
            f.write(json.dumps(eval(item)) + '\n')


if __name__ == '__main__':
    process("Amazon-531", "/mnt/disk5/hewenting_nfs_serverdir/tmp/output_file")