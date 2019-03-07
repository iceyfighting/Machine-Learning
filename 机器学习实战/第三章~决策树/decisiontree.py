import numpy as np
import pandas as pd
from collections import Counter
import math
from math import log

def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],
               ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    # 返回数据集和每个维度的名称
    return datasets, labels

datasets, labels = create_data()
train_data = pd.DataFrame(datasets, columns=labels)


#计算熵
def calc_ent(datasets):
    data_length = len(datasets)
    label_count = {}
    for i in range(data_length):
        label = datasets[i][-1]  #-1即颠倒过来  取标签值
        if label not in label_count:  #label_count {'否': 6, '是': 9} 若该标签值不在label_count这个字典中
            label_count[label] = 0    #令该标签值为0
        label_count[label] += 1       #令该标签树加一
    ent = -sum([(p/data_length)*log(p/data_length, 2) for p in label_count.values()]) #具体计算熵的公式
    return ent

# 经验条件熵
def cond_ent(datasets, axis=0):
    data_length = len(datasets)
    feature_sets = {}
    for i in range(data_length):
        feature = datasets[i][axis] #feature即每一行的第一个元素
        if feature not in feature_sets:
            #{'青年': [array(['青年', '否', '否', '一般', '否'], dtype='<U3'), array(['青年', '否', '否', '好', '否'], dtype='<U3')]}
            feature_sets[feature] = [] 
        feature_sets[feature].append(datasets[i]) #以每一行第一个元素作为键，每行元素作为值，组成键值对
    cond_ent = sum([(len(p)/data_length)*calc_ent(p) for p in feature_sets.values()]) #具体计算经验条件熵的公式
    return cond_ent

# 信息增益
def info_gain(ent, cond_ent):
    return ent - cond_ent

count = len(datasets[0]) - 1
ent = calc_ent(datasets) 
#print(ent)
best_feature = []
for c in range(count):
        c_info_gain = info_gain(ent, cond_ent(datasets, axis=c))  #每一行的信息增益
        best_feature.append((c, c_info_gain))
        print('特征({}) - info_gain - {:.3f}'.format(labels[c], c_info_gain))
#print(best_feature)
best_ = max(best_feature, key=lambda x: x[-1])
print(best_)
print('特征({})的信息增益最大，选择为根节点特征'.format(labels[best_[0]]))

