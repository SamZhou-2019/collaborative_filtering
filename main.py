from tqdm import tqdm

import util.datareader as reader
from util.evaluation import evaluation
from util.nearest_search import nearest
from util.item_rec import item_rec
from util.parse import parse_args

if __name__ == '__main__':
    args = parse_args()
    matrix, train_dict, test_dict = reader.read_matrix(args.dataset, args.type)
    n_users, n_items, train, test = matrix['users_num'], matrix['items_num'], \
        matrix['train_matrix'], matrix['test_matrix']

    nearest_n_users_dict, nearest_n_users_sim_dict = nearest(n_users, train, args.sim, int(args.nearest_user))

    result_file = open(args.rec_result,'w')

    users_rec_items = {}
    for user in tqdm(range(n_users), desc='item_rec', leave=True):
        rec_items, rec_items_score = item_rec(train, nearest_n_users_dict[user],
                                              nearest_n_users_sim_dict[user], train_dict, user, int(args.rec_item))
        users_rec_items[user] = rec_items
        result_file.write(str(user)+' '+str(rec_items)+'\n')

    p, r, f1 = evaluation(users_rec_items, test_dict)
    print("p\t%.4f\nr\t%.4f\nf1\t%.4f" % (p, r, f1))
