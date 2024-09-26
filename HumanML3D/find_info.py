import os
import numpy as np

def find_by_keyword(keyword, out_file): # keywork to search for among all motion descriptions
    all_files = []

    with open("all.txt", 'r') as f:
        for line in f.readlines():
            file = line.split('\n')[0]
            if os.path.exists(f"texts/{file}.txt"):
                all_files.append(file)

    interested_files = []
    for file_id in all_files:
        file = f"texts/{file_id}.txt"
        with open(file, 'r') as f:
            text = ''.join(f.readlines())
            if keyword in text:
                # data = np.load(f"new_joint_vecs/{file_id}.npy")
                # if len(data) < 199: continue
                interested_files.append(file_id)

    print("For keyword", keyword, len(interested_files), "samples found")

    with open(out_file, 'w') as f:
        for file in interested_files:
            print(file)
            f.write(f"{file}\n")

# find the most popular vocabulary in the HumanML3D dataset
def collect_stats():
    all_files = []
    with open("all.txt", 'r') as f:
        for line in f.readlines():
            file = line.split('\n')[0]
            if file.startswith('M'): continue
            if os.path.exists(f"texts/{file}.txt"):
                all_files.append(file)
    nouns = {}
    verbs = {}
    adjs = {}
    advs = {}
    props = {}
    for file_id in all_files:
        file = f"texts/{file_id}.txt"
        with open(file, 'r') as f:
            text = f.readlines()
            text = [s.split('#0.0')[0] for s in text]
            text = [s.split('#')[1] for s in text]
            text = [s.split() for s in text]
            text = [item for sublist in text for item in sublist]
            for token in text:
                if token.endswith('/NOUN'):
                    token = token.split('/NOUN')[0]
                    if token == 'zombie': print(file_id)
                    if token in nouns.keys(): nouns[token] += 1
                    else:                     nouns[token] = 1
                if token.endswith('/PROPN'):
                    token = token.split('/PROPN')[0]
                    if token == 'zombie': print(file_id)
                    if token in props.keys(): props[token] += 1
                    else:                     props[token] = 1
                if token.endswith('/VERB'):
                    token = token.split('/VERB')[0]
                    if token in verbs.keys(): verbs[token] += 1
                    else:                     verbs[token] = 1
                if token.endswith('/ADJ'):
                    token = token.split('/ADJ')[0]
                    if token in adjs.keys(): adjs[token] += 1
                    else:                     adjs[token] = 1
                if token.endswith('/ADV'):
                    token = token.split('/ADV')[0]
                    if token in advs.keys(): advs[token] += 1
                    else:                     advs[token] = 1

    # sort them
    nouns = dict(sorted(nouns.items(), key=lambda item: item[1], reverse=True))
    props = dict(sorted(props.items(), key=lambda item: item[1], reverse=True))
    verbs = dict(sorted(verbs.items(), key=lambda item: item[1], reverse=True))
    adjs = dict(sorted(adjs.items(), key=lambda item: item[1], reverse=True))
    advs = dict(sorted(advs.items(), key=lambda item: item[1], reverse=True))

    with open("nouns.txt", 'w') as f:
        for item in nouns.items():
            f.write(f"{item[0]:>12}  {item[1]:^6}\n")
    with open("props.txt", 'w') as f:
        for item in props.items():
            f.write(f"{item[0]:>12}  {item[1]:^6}\n")
    with open("verbs.txt", 'w') as f:
        for item in verbs.items():
            f.write(f"{item[0]:>12}  {item[1]:^6}\n")
    with open("adjs.txt", 'w') as f:
        for item in adjs.items():
            f.write(f"{item[0]:>12}  {item[1]:^6}\n")
    with open("advs.txt", 'w') as f:
        for item in advs.items():
            f.write(f"{item[0]:>12}  {item[1]:^6}\n")

def get_zombie_train(all):
    train = []
    non_train = []
    with open('zombie/zombie_found.txt', 'r') as f:
        lines2 = f.readlines()
        lines2 = [l.split('\n')[0] for l in lines2]
    for item in lines2:
        if item in all:
            train.append(item)
        else:
            non_train.append(item)
    return train, non_train

all = 29228
test_default = 4384
train_val_default = 24844
train_default = 23384
train_size = [
    11692,
    5846,
    2923,
    1461,
    730,
    365,
    182,
    91,
]

with open('train.txt', 'r') as f:
    lines = f.readlines()
    lines = [l.split('\n')[0] for l in lines]

zombie_train, zombie_non_train = get_zombie_train(lines)
other_train = list(set(lines) - set(zombie_train))

with open(f"zombie/zombie_train.txt", 'w') as f:
    for item in zombie_train:
        f.write(item+'\n')

for num_tot_train in train_size:
    num_non_zombie = num_tot_train - len(zombie_train)
    train_list = np.random.choice(other_train, num_non_zombie, replace=False).tolist()
    train_list += zombie_train
    assert len(train_list) == num_tot_train
    with open(f"train_{num_tot_train}.txt", 'w') as f:
        for s in train_list:
            f.write(s+'\n')



