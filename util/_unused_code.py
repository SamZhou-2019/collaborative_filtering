# 这里保留了一些测试代码和不再使用的代码

# \util\nearest_search.py
def nearest(n_users, matrix, sim_type='c', N=5):
    """
    寻找与每个用户最相似的N个用户，以及相似度。
    方法是构造N+1的列表，当相似度大于列表中的值时，将其放入列表中的值的位置，并将后续值后移。最终保留前N项。
    """
    nearest_n_users_dict = {}
    nearest_n_users_sim_dict = {}
    for target_user in tqdm(range(n_users), desc='total', leave=True):
        nearest_n_users = [-1] * (N + 1)
        nearest_n_users_sim = [-1] * (N + 1)
        # for other_user in tqdm(range(n_users), desc='target_user:'+str(target_user), leave=True):
        for other_user in range(n_users):
            if other_user != target_user:
                users_sim = sim_calc(matrix[target_user].toarray().tolist()[0],
                                     matrix[other_user].toarray().tolist()[0], sim_type)

                for i in range(N):
                    if users_sim > nearest_n_users_sim[i]:
                        nearest_n_users_sim[i+1:N + 1] = nearest_n_users_sim[i:N]
                        nearest_n_users_sim[i] = users_sim
                        nearest_n_users[i+1:N + 1] = nearest_n_users[i:N]
                        nearest_n_users[i] = other_user
                        break

        nearest_n_users_sim_dict[target_user] = nearest_n_users_sim[:N]
        nearest_n_users_dict[target_user] = nearest_n_users[:N]

    return nearest_n_users_dict, nearest_n_users_sim_dict


def nearest_v2(n_users, matrix, dok_matrix, sim_type='c', N=5):
    """
    寻找与每个用户最相似的N个用户，以及相似度。
    方法是构造键为用户序号、值为相似度的字典。
    计算目标用户与其它用户的相似度并保存，对字典按值排序，保留前N个用户序号及相似度。
    注意“其它用户”不包含目标用户

    这个函数的运行时间比nearest快三分之一。
    """
    nearest_n_users_dict = {}
    nearest_n_users_sim_dict = {}
    for target_user in tqdm(range(n_users)[0:1], desc='total', leave=True):
        users_sim = {}

        for other_user in range(n_users):
            if other_user != target_user:
                sim = sim_calc(matrix[target_user].tolist()[0],
                               matrix[other_user].tolist()[0],
                               dok_matrix[target_user],
                               dok_matrix[other_user],
                               sim_type)
                users_sim[other_user] = sim

        sorted_users_sim = sorted(users_sim.items(), key=lambda e: e[1], reverse=True)
        nearest_n_users_sim_dict[target_user], nearest_n_users_dict[target_user] = \
            [sorted_users_sim[i][1] for i in range(N)], [sorted_users_sim[i][0] for i in range(N)]

    return nearest_n_users_dict, nearest_n_users_sim_dict

# \util\similarity.py
def sim_calc(list1: list, list2: list, dok1, dok2, sim_type='c'):
    """
    对两个用户的相似度进行计算
    :param list1: 用户1的交互项目
    :param list2: 用户2的交互项目
    :param sim_type: 相似度类型：{c:余弦相似度（默认）;t:tanimoto相似度;l:对数似然比;m:曼哈顿距离}
    :return: 指定的相似度
    """
    assert len(list1) == len(list2), "Error: length of two lists dont equal."
    # list1_no, list2_no, intersect_no = 0.0, 0.0, 0.0

    list1_no = dok1.getnnz()
    list2_no = dok2.getnnz()
    intersect_no = (list1_no + list2_no) - np.add(dok1, dok2).getnnz()
    print('Calculating similarity.')
    '''
    for i in range(len(list1)):
        if list1[i] != 0.0:  # 用户1交互过的项目数量
            list1_no += 1.
        if list2[i] != 0.0:  # 用户2交互过的项目数量
            list2_no += 1.
        if list1[i] != 0.0 and list2[i] != 0.0:  # 两个用户交互过的项目交集数量
            intersect_no += 1.
    '''
    if sim_type == 'c':
        sim = 1.0 - spatial.distance.cosine(list1, list2) if intersect_no != 0 else 0
    elif sim_type == 't':
        # tanimoto相似度计算方法：列表不为0部分的交集长度除以并集长度
        sim = intersect_no / (list1_no + list2_no + intersect_no) if intersect_no != 0 else 0
    elif sim_type == 'l':
        sim = loglikelihood(intersect_no, list1_no - intersect_no,
                            list2_no - intersect_no, len(list1) - list1_no - list2_no + intersect_no)
    elif sim_type == 'm':
        # cityblock距离也叫曼哈顿距离，原理是两个点在每个坐标上的差的绝对值之和
        sim = 0.0
        for i in range(len(list1)):
            sim += math.fabs(float(list1[i] - list2[i]))
    else:
        raise "Error: unexpected sim_type."
    print('Similarity calculated completely.')

    return sim


def loglikelihood_old(k11, k12, k21, k22):
    """
    LLR，即对数似然比。对数似然比LLR用于度量两个事件的关联性
    设：两个事件同时发生的次数 k11
    一个事件发生而另外一个事件没有发生的次数 k12和k21
    两个事件都没有发生 k22
    则计算公式为LLR=2*N*(RowEntropy+ColumnEntropy-MatrixEntropy)
    其中RowEntropy为每行信息熵之和
    ColumnEntropy为每列信息熵之和
    MatrixEntropy为整个矩阵信息熵之和
    计算代码如下。
    以上计算过程参考了：
    https://nikaashpuri.wordpress.com/2016/03/09/llr-log-likelihood-ratio-used-for-recommendations/
    """
    N = sum([k11, k12, k21, k22])
    row_sum = xLogX((k11 + k12) / N) + xLogX((k21 + k22) / N)
    col_sum = xLogX((k11 + k21) / N) + xLogX((k12 + k22) / N)
    mat_sum = xLogX(k11 / N) + xLogX(k21 / N) + xLogX(k12 / N) + xLogX(k22 / N)

    LLR = 2.0 * N * (mat_sum - row_sum - col_sum)

    if LLR > 0:
        return LLR
    else:
        return 0.0


def xLogX_old(x):
    # 该函数用于辅助计算LLR，其结果是返回x·log(x)，即信息熵的负数
    return 0.0 if x == 0 else x * math.log(x)

