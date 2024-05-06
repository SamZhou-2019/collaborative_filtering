from util.similarity import sim_calc
from tqdm import tqdm


def nearest(n_users, matrix, sim_type='c', N=5):
    """
    寻找与每个用户最相似的N个用户，以及相似度。
    方法是构造键为用户序号、值为相似度的字典。
    计算目标用户与其它用户的相似度并保存，对字典按值排序，保留前N个用户序号及相似度。
    注意“其它用户”不包含目标用户
    """
    nearest_n_users_dict = {}
    nearest_n_users_sim_dict = {}
    sim_matrix = sim_calc(matrix.todense(), sim_type)
    for target_user in tqdm(range(n_users), desc='nearest_search', leave=True):
        users_sim = {}
        for other_user in range(n_users):
            if other_user != target_user:
                users_sim[other_user] = sim_matrix[target_user][other_user]

        sorted_users_sim = sorted(users_sim.items(), key=lambda e: e[1], reverse=True)
        nearest_n_users_sim_dict[target_user], nearest_n_users_dict[target_user] = \
            [sorted_users_sim[i][1] for i in range(N)], [sorted_users_sim[i][0] for i in range(N)]

    return nearest_n_users_dict, nearest_n_users_sim_dict