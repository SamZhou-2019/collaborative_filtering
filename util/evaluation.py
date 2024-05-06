import numpy as np
from tqdm import tqdm


def evaluation(users_rec_items, test_dict):
    users_p, users_r, users_f1 = [], [], []
    for target_user in tqdm(users_rec_items.keys(), desc='evaluation', leave=True):
        intersect_set = set(users_rec_items[target_user]).intersection(test_dict[target_user])
        p = len(list(intersect_set)) / len(users_rec_items[target_user])
        r = len(list(intersect_set)) / len(test_dict[target_user])
        f1 = (2.0 * p * r) / (p + r) if p + r > 0 else 0

        users_p.append(p)
        users_r.append(r)
        users_f1.append(f1)

    return np.mean(users_p), np.mean(users_r), np.mean(users_f1)
