import torch
import numpy as np

def loglikelihood(k11, k12, k21, k22):
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
    LLR = torch.where(LLR < 0, torch.tensor(0), LLR)

    return LLR


def xLogX(x):
    # 该函数用于辅助计算LLR，其结果是返回x·log(x)，即信息熵的负数
    return torch.where(x == 0, torch.tensor(0), x * torch.log(x.float()))

def sim_calc(matrix, sim_type='c'):
    """
    对用户之间的相似度进行计算
    :param matrix: 用户与项目的交互矩阵
    :param sim_type: 相似度类型：{c:余弦相似度（默认）;t:tanimoto相似度;l:对数似然比;m:曼哈顿距离}
    :return: 指定的相似度
    """
    print('Calculating similarity.')
    if sim_type == 'c':
        sim = matrix_cos_sim(matrix) if torch.cuda.is_available() else matrix_cos_sim_cpu(matrix)
    else:
        matrix_tensor = torch.tensor(matrix, device='cuda') if torch.cuda.is_available() \
            else torch.tensor(matrix)
        non_zero_count = torch.count_nonzero(matrix_tensor, dim=1)
        union_matrix = \
            torch.count_nonzero((matrix_tensor.unsqueeze(0) + matrix_tensor.unsqueeze(1)), dim=2)
        if sim_type == 't':
            # tanimoto相似度计算方法：列表不为0部分的交集长度除以并集长度
            sim = union_matrix / (non_zero_count + non_zero_count.transpose(-1, 0) - union_matrix)
            sim = sim.cpu()
        elif sim_type == 'l':
            sim = loglikelihood(union_matrix, non_zero_count - union_matrix,
                                non_zero_count.transpose(-1, 0) - union_matrix, matrix_tensor.shape[0] - union_matrix)
            sim = sim.cpu()
        elif sim_type == 'm':
            # cityblock距离也叫曼哈顿距离，原理是两个点在每个坐标上的差的绝对值之和
            sim = torch.cumsum(matrix_tensor.unsqueeze(0)-matrix_tensor.unsqueeze(1), dim=2)
            torch.abs_(sim)
        else:
            raise "Error: unexpected sim_type."

    print('Similarity calculated completely.')
    return sim


def matrix_cos_sim(matrix):
    """
    求矩阵中每行与其他行的余弦相似度，最后返回一个行数x行数的矩阵
    """
    matrix_tensor = torch.tensor(matrix, device='cuda')
    row_norms = torch.norm(matrix_tensor, dim=1)
    dot_matrix = torch.matmul(matrix_tensor, matrix_tensor.t())
    dot_row_norms = 1 / torch.matmul(row_norms, row_norms.t())
    cosine_sim_matrix = dot_matrix.mul_(dot_row_norms)
    return cosine_sim_matrix.cpu().numpy()


def matrix_cos_sim_cpu(matrix):
    """
    求矩阵中每行与其他行的余弦相似度，最后返回一个行数x行数的矩阵
    """
    row_norms = np.linalg.norm(matrix, axis=1)
    dot_matrix = np.dot(matrix, np.transpose(matrix))
    dot_row_norms = np.dot(row_norms, np.transpose(row_norms))
    cosine_sim_matrix = dot_matrix / dot_row_norms
    return cosine_sim_matrix
