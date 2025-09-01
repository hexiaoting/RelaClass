from argparse import ArgumentParser
from tqdm import tqdm

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str, required=False, default="/home/hewenting/data_preprocess/Amazon-531/Amazon-531_train.json")
    parser.add_argument('--label2name_file', type=str, required=False, default='/mnt/disk5/hewenting_nfs_serverdir/datasets/Amazon-531/train/labels.txt')
    args = parser.parse_args()
    return args

def print_tree(class_tree, id2name):
    class_tree["top"].sort(key=abs)
    for i in class_tree["top"]:
        print(f'{i}-{id2name[i]} ({len(class_tree[i])}):')
        class_tree[i].sort()
        for j in class_tree[i]:
            print(f'\t{j}-{id2name[j]} ({len(class_tree[j])}):')
            class_tree[j].sort()
            for k in class_tree[j]:
                print(f'\t\t{k}-{id2name[k]}')

def print_class_level_0_1(class_tree, id2name, level = 0):
    class_tree["top"].sort(key=abs)
    for i in class_tree["top"]:
        print(id2name[i], end=' : ')
        class_tree[i].sort()
        for j in class_tree[i]:
            print(id2name[j], end=',')
        print("")

def print_class(class_tree, id2name, level = 0):
    class_tree["top"].sort(key=abs)
    for i in class_tree["top"]:
        print(id2name[i], end=' : ')
        class_tree[i].sort()
        for j in class_tree[i]:
            print(id2name[j], "(", end='')
            class_tree[j].sort()
            for k in class_tree[j]:
                print(id2name[k], end=' ')
            print("); ", end='')
        print("")
        # break
                      

def process(args):
    id2name = {}
    with open(args.label2name_file) as f:
        for line in f.readlines():
            line = line.strip('\n').split('\t')
            id2name[int(line[0])] = line[1]

    class_tree = {"top":[]}
    with open(args.input_file) as f:
        readin = f.readlines()
        for line in tqdm(readin):
            item = eval(line)
            label = item['label']
            for i, id in enumerate(label):
                if i == 0:
                    if id not in class_tree["top"]:
                        class_tree["top"].append(id)
                if i < 2 and id not in class_tree.keys():
                    class_tree[id] = []
                if i > 0 and id not in class_tree[label[i-1]]:
                    class_tree[label[i-1]].append(id)

    print_tree(class_tree, id2name)
    #print_class(class_tree, id2name)
    #print_class_level_0_1(class_tree, id2name)


if __name__ == '__main__':
    args = parse_args()
    process(args)
