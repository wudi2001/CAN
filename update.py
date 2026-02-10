import torch
from torch.nn.functional import softmax
from collections import Counter


community_num = 7  # 具有七个社区
cortex_node_num = 210  # 使用的脑图谱包含210个皮层节点

def init_freq():    # 每轮epoch结束，重新初始化freq
    freq = {}
    for i in range(cortex_node_num):
        dic = dict()
        for j in range(community_num):
            dic[j] = 0
        freq[i] = dic
    freq["num"] = 0   # 记录社区分配总数
    return freq

def get_community_centroid(community_label, node_feature):  # 获取社区质心同时获取超参数tau  该函数每来一批数据都要进行一次计算，但至于是否每个样本都要计算各自的质心需要进行实验
    community_centroid = {}  # 存储社区质心
    node_feature = node_feature.permute(2, 0, 1, 3)  # 原本维度是N'*T*N*d，转为N*N'*T*d，原本第一个维度是该类别下的样本总数
    for community, node_list in community_label.items():  #获取社区编号以及对应社区下的节点编号列表
        community_node_feature = node_feature[node_list] # 获取对应社区下的所有节点特征
        community_node_feature = community_node_feature.reshape(-1, community_node_feature.shape[-1])
        community_centroid[community] = torch.mean(community_node_feature, dim=0)
    return community_centroid

def count_freq(community_centroid, node_feature, freq, tau, beta, update):   # 其中的beta为指数平均中的衰减率
    community_number = len(community_centroid)
    embedding_dimension = node_feature.shape[-1]

    orthogonal_community_centroid = torch.zeros(   #得到对应的正交化后的社区质心
        community_number, embedding_dimension, dtype=torch.float
    )
    orthogonal_community_centroid[0] = community_centroid[0] / torch.norm(community_centroid[0], p=2)
    for i in range(1, community_number):
        project = 0
        for j in range(i):
            u = community_centroid[j]
            v = community_centroid[i]
            project += (torch.dot(u, v) / torch.dot(u, u)) * u

        community_centroid[i] -= project
        orthogonal_community_centroid[i] = community_centroid[i] / torch.norm(community_centroid[i], p=2)

    # 执行社区分配操作
    node_feature = node_feature.permute(2, 0, 1, 3)
    # 社区分配只针对大脑皮层节点进行
    cortex_node_feature = node_feature[:cortex_node_num]
    cortex_node_feature = cortex_node_feature.reshape(cortex_node_num, -1, cortex_node_feature.shape[-1])  #将中间两个维度进行组合
    if update:   # 只有社区更新时才统计
        freq["num"] += cortex_node_feature.shape[1]
    order_node = 0  #记录当前处理节点的次序
    for node in cortex_node_feature:   # 对每个节点进行遍历
        norm = torch.norm(node, p=2, dim=-1, keepdim=True)
        node = node / norm   #对节点特征进行归一化
        dot_data = torch.mm(node, orthogonal_community_centroid.T)
        prob = softmax(dot_data, dim=-1)

        top2_value, top2_idx = torch.topk(prob, 2, dim=-1)
        first_value, first_idx = top2_value[:, 0], top2_idx[:, 0]   # 获取最大的概率
        second_value, second_idx = top2_value[:, 1], top2_idx[:, 1]
        mean_value = torch.mean(first_value - second_value)
        tau = beta * tau + (1 - beta) * mean_value  # 采用指数平均的方式更新阈值tau

        if update:
            bool_first_value = (first_value - second_value) > tau
            idx_list = first_idx[bool_first_value]    # 获取该节点对应的各个社区的索引
            idx_list = idx_list.numpy()
            count = Counter(idx_list)
            #freq[order_node] = dict(Counter(freq[order_node]) + count)
            freq[order_node].update(count)
            order_node += 1

    return freq, tau


def update_community(freq, threshold=0.06):  #每次epoch结束进行一次更新
    updated_community_label = {}
    for num in range(community_num):
        updated_community_label[num] = []
    num = freq.pop("num")
    for node, community_freq in freq.items():
        highest_freq_community = max(community_freq, key=community_freq.get)
        highest_freq_value = community_freq[highest_freq_community]  # 获取最大频次
        if highest_freq_value / num > threshold:
            updated_community_label[highest_freq_community].append(node)

    return updated_community_label

# 注意以下脑区编号都是从0开始计数，后续画图时要全部加1
community_label = {
    0: [104, 105, 113, 118, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 201, 202, 203, 204, 205, 206, 207],
    1: [52, 56, 58, 66, 67, 70, 71, 72, 75, 154, 155, 156, 157, 160, 161],
    2: [97, 124, 125, 126, 127, 132, 133],
    3: [37, 168, 172, 182],
    4: [46, 49, 68, 69, 89, 92, 93, 108, 109, 116],
    5: [20, 21, 23, 30, 31],
    6: [4, 5, 13, 32, 41, 42, 86, 87, 94, 152, 174, 186, 187]}

# node_feature = torch.randn([4, 20, 279, 64])
#
# community_centroid = get_community_centroid(community_label, node_feature)
#
# count_freq(community_centroid, node_feature, freq, 0.5, 0.9, False)
#
# node_feature1 = torch.randn([10, 20, 279, 64])
# community_centroid1 = get_community_centroid(community_label, node_feature1)
# count_freq(community_centroid1, node_feature1, freq, 0.01, 0.9, True)
# updated_community_label = update_community(freq)