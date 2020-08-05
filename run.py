import time
import torch
import numpy as np
from test_pytorch import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif, device
from attention_pytorch import *
import pickle

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--embedding', default='random', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

embed_dim = 128
class_list = [x.strip() for x in open(
            './.data/class.txt', encoding='utf-8').readlines()]
num_class = len(class_list)

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(args.word)
    train_iter = build_iterator(train_data)
    dev_iter = build_iterator(dev_data)
    test_iter = build_iterator(test_data)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    vocab_size = len(pickle.load(open('./.data/vocab.pkl', 'rb')))
    # train
    #model = TextCNN_attn(vocab_size, embed_dim, num_class, device).to(device)
    model = TextCNN(vocab_size, embed_dim, num_class).to(device)
    init_network(model)
    print(model.parameters)
    train(model, train_iter, dev_iter, test_iter)