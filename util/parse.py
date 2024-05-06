import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='协同过滤')
    parser.add_argument('--dataset', nargs='?', default='data', help='存放数据的文件夹，其中包含train.txt和test.txt。')
    parser.add_argument('--type', nargs='?', default='1toA',
                        help='数据的保存类型，分为：\n'
                             '1to1:每行仅有两个元素，第一个为用户编号，第二个为项目编号；\n'
                             '1to1_score:每行仅有三个元素，第一个为用户编号，第二个为项目编号，第三个为权重；\n'
                             '1toA:每行包含多个元素，第一个为用户编号，其余为项目编号（默认）。')
    parser.add_argument('--sim', nargs='?', default='c', help='相似度计算方式，包括：'
                                                              '{c:余弦相似度（默认）;t:tanimoto相似度;l:对数似然比;m:曼哈顿距离}')
    parser.add_argument('--nearest_user', nargs='?', default='10', help='为项目评分时参考的相似用户的数量')
    parser.add_argument('--rec_item', nargs='?', default='100', help='最终推荐的项目数量')
    parser.add_argument('--rec_result', nargs='?', default='rec_result.txt', help='最终推荐的项目结果保存路径（文本文件）')
    return parser.parse_args()
