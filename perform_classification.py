
import numpy as np
import torch as th

from torch import nn
import torch.utils
import torch.utils.data
import torch.utils.data.sampler
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import f1_score

#dico = th.load("wordnet_nouns_0_01_with_normalized_rank.pth")
dico = th.load("wordnet_nouns_0_01.pth")


model = dico['model']
epoch = dico['epoch']
objects = dico['objects']

model = model['lt.weight']

name_roots = ['animal.n.01', 'group.n.01', 'worker.n.01', 'mammal.n.01']
nb_runs = 5


for name in name_roots:
    f1_scores = []
    name_root = name.split('.')
    name_root = name_root[0]
    beta = 1
    for run in range(nb_runs):
        print("split %d of subtree %s" % (run+1, name))
        print("loading training examples")
        train_positive = []
        train_negative = []
        train_label_positive = []
        train_label_negative = []
        f = open("train_%s_%d.txt" % (name_root, run+1))
        for line in f:
            l = line.split(" ")
            if "1" in l[0]:
                train_positive.append(objects.index(l[1].strip()))
                train_label_positive.append(1)
            else:
                train_negative.append(objects.index(l[1].strip()))
                train_label_negative.append(0)

        f.close()
        trn = th.LongTensor(train_negative)
        trp = th.LongTensor(train_positive)
        trpn = th.cat((trp, trn),0)
        tensor_data = model[trpn]

        trln = th.LongTensor(train_label_negative)
        trlp = th.LongTensor(train_label_positive)
        tensor_target = th.cat((trlp, trln),0)
        nb_train_pos = len(train_positive)
        nb_train_neg = len(train_negative)

        print("classifying test examples")

        u0 = -(th.sqrt(th.pow(tensor_data,2).sum(-1, keepdim=True) + beta))
        u = th.cat((tensor_data,u0),-1)
        

        true_labels = []
        predicted_labels = []
        test_positive = []
        test_negative = []
        f = open("test_%s_%d.txt" % (name_root, run+1))
        cpt = 0
        for line in f:
            l = line.split(" ")
            index_ = th.LongTensor([objects.index(l[1].strip())])
            v = model[index_]
            v0 = th.sqrt(th.pow(v,2).sum(-1, keepdim=True) + beta)
            v = th.cat((v,v0),-1).expand_as(u)
            e = th.sum(u * v, dim=-1)

            (value, max_index) = th.max(e, -1)

            if "1" in l[0]:
                true_labels.append(1)
            else:
                true_labels.append(0)

            if max_index.numpy()[0] < nb_train_pos:
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)

        f.close()
        score = f1_score(true_labels, predicted_labels)
        print("f1 score for the current split: %f percents" % (score * 100))
        f1_scores.append(score * 100)
        

    np_scores = np.array(f1_scores)
    print("f1 score results for the subtree %s: mean: %f, std: %f" % (name, np.mean(np_scores), np.std(np_scores, ddof=1)))
    
