import os,sys,json
from pprint import pprint
# import ast
from collections import OrderedDict

ip_fname = '../data/Lu_Justin_dataset/reviews_TV.txt'
op_fname = ip_fname.replace('.txt','.json')
lines = [l.strip() for l in open (ip_fname)]
d = {}
to_split_keys = ['pros_raw_annotation','pros_feature','pros_aspect',
                 'cons_raw_annotation','cons_feature','cons_aspect']
for l in lines:
    if l.startswith('id:'):
        id = int(l.split()[-1])
        d[id] = OrderedDict()
        d[id] = {}
        continue
    else:
        k = l.split(': ')[0]
        if not k: continue
        v = l.split(': ')[-1]
        if v.startswith('['):
            newv = []
            for vv in v[1:-1].split(', '):
                try:
                    newv.append(vv.encode('utf-8'))
                except:
                    pass
            v = newv
        if k in to_split_keys:
            v = [vv.strip().encode('utf-8') for vv in v.split(';')]
        d[id][k] = v

with open(op_fname,'w') as fh:
   json.dump(d,fh,indent=4)

pprint (d)

# print 'please check {} for the json dict output'.format(op_fname)

