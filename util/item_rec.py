import numpy as np


def item_rec(matrix, nearest_n_users, nearest_n_users_sim, train_dict, target_user, rec_item_num=100):
    """
    根据目标用户的最相似的N个用户，以及相似度，获取推荐的项目
    原理是将N个用户的相似度归一化之后，将这N个用户在矩阵中的行按相似度进行加权求和，得到目标用户对全部项目的评分
    排除目标用户在训练集中已经交互过的项目，取得评分前m的项目，作为该目标用户的预测结果。m=rec_item_num
    """
    weight_sim = np.divide(nearest_n_users_sim, sum(nearest_n_users_sim)).tolist()
    items_score, items_score_dict = [0], {}

    matrix = matrix.todense()
    for i, user in enumerate(nearest_n_users):
        user_weight_sim = weight_sim[i]
        user_row = matrix[user].tolist()[0]
        items_score = np.add(items_score, user_weight_sim * np.array(user_row, dtype=np.float64)).tolist()

    for i, item_score in enumerate(items_score):
        # 注意预测的结果中不包含 目标用户在训练集中的项目
        if i not in train_dict[target_user]:
            items_score_dict[i] = item_score

    sorted_items_score_dict = sorted(items_score_dict.items(), key=lambda a:a[1], reverse=True)
    rec_items, rec_items_score = \
        [sorted_items_score_dict[i][0] for i in range(rec_item_num)], \
        [sorted_items_score_dict[i][1] for i in range(rec_item_num)]

    return rec_items, rec_items_score


