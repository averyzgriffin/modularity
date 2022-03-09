import numpy as np
from os.path import join
from os import replace
import json
from tqdm import tqdm

# example = [0, 1, 0, 0,
#           1, 1, 0, 0]

def default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError('Not serializable')


def load_samples(num_samples, dir):
    loaded_samples = []
    for i in tqdm(range(num_samples), desc="Loading labeled data"):
        w_file = open(join(dir, f"sample_{i}.json").replace("\\", "/"), "r")
        sample = json.load(w_file)
        sample["pixels"] = np.array(sample["pixels"])
        loaded_samples.append(sample)
        w_file.close()
    return loaded_samples


def save_samples(samples, name):
    for i in range(len(samples)):
        w_file = open(f"samples/{name}/sample_{i}.json", "w")
        json.dump(samples[i], w_file, default=default)
        w_file.close()


def label_sample(sample):
    left = False
    right = False
    if sum(sample["pixels"][0:4]) >= 1 or sum(sample["pixels"][0:2]) >= 1:
        left = True
    if sum(sample["pixels"][4:8]) >= 1 or sum(sample["pixels"][6:8]) >= 1:
        right = True

    if left and right:
        sample["str_label"] = "both"
        sample["int_label"] = 3
    elif left:
        sample["str_label"] = "left"
        sample["int_label"] = 1
    elif right:
        sample["str_label"] = "right"
        sample["int_label"] = 2
    else:
        sample["str_label"] = "neither"
        sample["int_label"] = 0

    return sample


def generate_samples(n):
    all_samples = []
    all_arrays = []
    both_count = 0
    left_count = 0
    right_count = 0
    neither_count = 0

    while len(all_samples) < n:
        random_sample = {"pixels": np.array(np.random.randint(2, size=8)), "str_label": "tbd", "int_label": 0}
        random_sample = label_sample(random_sample)

        if random_sample["pixels"].tolist() in all_arrays:
            continue
        else:
            if random_sample["int_label"] == 3:
                if both_count < 42:
                    both_count += 1
                    all_samples.append(random_sample)
                    all_arrays.append(random_sample["pixels"].tolist())
            elif random_sample["int_label"] == 1:
                if left_count < 21:
                    left_count += 1
                    all_samples.append(random_sample)
                    all_arrays.append(random_sample["pixels"].tolist())
            elif random_sample["int_label"] == 2:
                if right_count < 21:
                    right_count += 1
                    all_samples.append(random_sample)
                    all_arrays.append(random_sample["pixels"].tolist())
            elif random_sample["int_label"] == 0:
                if True: #neither_count <= left_count and neither_count <= right_count and neither_count <= both_count:
                    neither_count += 1
                    all_samples.append(random_sample)
                    all_arrays.append(random_sample["pixels"].tolist())

    return all_samples


def filter_samples(samples, labels: list):
    filtered_samples = []
    for sample in samples:
        for label in labels:
            if sample["int_label"] == label:
                filtered_samples.append(sample)
    return filtered_samples

# all_samples = []
# for i in range(1000):
#     random_sample = {"pixels": np.array(np.random.randint(2, size=8)), "str_label": "tbd", "int_label": 0}
#     random_sample = label_sample(random_sample)
#     print(random_sample, "\n")
#     all_samples.append(random_sample)


