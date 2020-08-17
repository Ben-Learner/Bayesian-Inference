#!/usr/bin/env python
# -*- coding:utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import pandas as pd
import numpy as np
from pgmpy.inference import VariableElimination
#建立模型
LLOCA_model = BayesianModel()
model = pd.read_excel('model.xlsx','LLOCA子节点')
model_df = pd.DataFrame(model)


for i in range(model_df.shape[0]):
    for j in range(1,model_df.shape[1]):
        if model_df.iat[i,j] is not np.nan:
            LLOCA_model.add_edge(str(model_df.iat[i,0]),str(model_df.iat[i,j]))
plt.rcParams['font.sans-serif'] = ['SimHei']
nx.draw_circular(LLOCA_model,  with_labels=True, arrowsize=30, node_size=800, alpha=0.3, font_weight='bold')
plt.savefig('LLOCA_model.png', bbox_inches='tight')

# lloca_model = BayesianModel([('LLOCA','1#冷却剂压力'),
#                              ('LLOCA','2#冷却剂压力'),
#                              ('LLOCA','安全壳放射性'),
#                              ('LLOCA','安全壳压力'),
#                              ('LLOCA','安全壳温度'),
#                              ('LLOCA','地坑水位'),
#                              ('LLOCA','1#冷却剂流量'),
#                              ('LLOCA','2#冷却剂流量'),
#                              ('1#冷却剂压力','一回路平均压力'),
#                              ('2#冷却剂压力', '一回路平均压力'),
#                              ('一回路平均压力', '稳压器压力'),
#                              ('稳压器压力', '稳压器水位')
#                              ])

#添加CPD
parent_node = pd.read_excel('model.xlsx',sheet_name='LLOCA父节点')
parent_node_df = pd.DataFrame(parent_node)
CPD = pd.read_excel('model.xlsx',sheet_name='节点状态数及CPD')
CPD_df = pd.DataFrame(CPD)
all_cpd_list = []
all_evidence_list = []
all_evidence_card_list = []
for i in range(CPD_df.shape[0]):
    cpd_list = []
    if parent_node_df.iat[i,1] is np.nan:
        for j in range(CPD_df.iat[i,1]):
            sub_cpd_list = []
            sub_cpd_list.append(CPD_df.iat[i,2+j])
            cpd_list.append(sub_cpd_list)
    else:
        for k in range(CPD_df.shape[0]):
            if CPD_df.iat[k,0] == parent_node_df.iat[i,1]:
                parent_status_count = int(CPD_df.iat[k,1])
            if CPD_df.iat[k,0] == parent_node_df.iat[i,0]:
                self_status_count = int(CPD_df.iat[k,1])
        # print(self_status_count)
        # print(parent_status_count)
        for m in range(self_status_count):
            sub_cpd_list = []
            for l in range(parent_status_count):
                sub_cpd_list.append((CPD_df.iat[i,2 + m + l  * self_status_count]))
            cpd_list.append(sub_cpd_list)
    all_cpd_list.append(cpd_list)
# print(all_cpd_list)

for i in range(parent_node_df.shape[0]):
    evidence_list = []
    evidence_card_list = []
    for j in range(1,parent_node_df.shape[1]):
        if parent_node_df.iat[i, j] is np.nan:
            evidence_list.append('nan')
        else:
            evidence_list.append(str(parent_node_df.iat[i,j]))
    all_evidence_list.append(evidence_list)
    if evidence_list[0] == 'nan':
        evidence_card_list.append('nan')
    else:
        for k in range(len(evidence_list)):
            for l in range(CPD_df.shape[0]):
                if evidence_list[k] == CPD_df.iat[l,0]:
                    evidence_card_list.append(int(CPD_df.iat[l,1]))
    all_evidence_card_list.append(evidence_card_list)
# print(all_evidence_card_list)
# print(all_evidence_list)
for i in range(CPD_df.shape[0]):
    if all_evidence_list[i] == ['nan']:
        evidence1 = []
        evidence_card1 = []
    else:
        evidence1 = all_evidence_list[i]
        evidence_card1 = all_evidence_card_list[i]
    # print(evidence1)
    # print(evidence_card1)
    cpd_i = TabularCPD(str(CPD_df.iat[i,0]), int(CPD_df.iat[i,1]), all_cpd_list[i], evidence = evidence1, evidence_card = evidence_card1)
    LLOCA_model.add_cpds(cpd_i)
# print(LLOCA_model.check_model())
# print(LLOCA_model.get_cpds('稳压器压力').values)
LLOCA_infer = VariableElimination(LLOCA_model)
q = LLOCA_infer.query(variables=['LLOCA'],evidence={'稳压器水位':0,'安全壳压力':1},joint=False)
print(q['LLOCA'])






