import random

name_roots = ['animal.n.01', 'group.n.01', 'worker.n.01', 'mammal.n.01']
training = [3218,6649,861,953]
test = [798,1727,254,228]

nb_runs = 5

index_name = -1
for name in name_roots:
    index_name += 1
    cpt = 0
    name_root = name.split('.')
    name_root = name_root[0]
    positive = {}
    negative = {}
    f = open('noun_closure.tsv', 'r')
    for line in f:
        l = line.split('\t')
        if name in l[1]:
            cpt += 1
            positive[l[0]] = 0            
    f.close()

    f = open('noun_closure.tsv', 'r')
    for line in f:
        l = line.split('\t')
        if l[0] in name:
             continue 
        if l[0] not in positive:
            negative[l[0]] = 0

    f.close()

    print('%s %d %d' % (name, cpt, training[index_name] + test[index_name]))

    for run in range(nb_runs):
        name_root = name.split('.')
        name_root = name_root[0]
        pos = positive.keys()
        random.shuffle(pos)
        postr = pos[0:training[index_name]]
        poste = pos[training[index_name]:(training[index_name] + test[index_name])]
        neg = negative.keys()
        nb_neg = len(neg)
        nb_neg_train = int(0.8 * len(neg))
        
        random.shuffle(neg)
        negtr = neg[0:nb_neg_train]
        negte = neg[nb_neg_train:nb_neg]
        
        ftr = open("train_%s_%d.txt" % (name_root, run+1), "w")
        fte = open("test_%s_%d.txt" % (name_root, run+1), "w")
        for tr in postr:
            ftr.write("1 %s\n" % tr)
        for te in poste:
            fte.write("1 %s\n" % te)
        for tr in negtr:
            ftr.write("0 %s\n" % tr)
        for te in negte:
            fte.write("0 %s\n" % te)
    
        ftr.close()
        fte.close()

