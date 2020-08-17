from pgmpy.readwrite import BIFReader
reader = BIFReader('低压安注直接注入.bif')
LLOCA_model = reader.get_model()
# print(LLOCA_model.nodes)
# print(LLOCA_model.edges)
# LLOCA_model.get_cpds()
# for cpd in LLOCA_model.get_cpds():
#     print('CPD of {variable}:'.format(variable=cpd.variable))
#     print(cpd)
# print(LLOCA_model.get_independencies())
# print(LLOCA_model.check_model())
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt
nx.draw_circular(LLOCA_model,  with_labels=True, arrowsize=15, node_size=2000, font_size=8, alpha=0.3, font_weight='bold')
plt.savefig('LLOCA_model1.png', bbox_inches='tight')
LLOCA_infer = VariableElimination(LLOCA_model)
a = LLOCA_infer.query(variables=['LHSI-DI'],evidence={'CP':'higher','PTR001BA-WL':'higher'},joint=False)
print(a['LHSI-DI'])
##添加节点
# from pgmpy.factors.discrete import TabularCPD
# LLOCA_model.add_edge('B','LLOCA')
# B_cpd = TabularCPD('B', 3, [[0.6,0.2],[0.3,0.5],[0.1,0.3]], evidence=['LLOCA'], evidence_card=[2])
# LLOCA_model.add_cpds(B_cpd)
# from pgmpy.readwrite import BIFWriter
# writer = BIFWriter(model=LLOCA_model)
# print(writer)
##删去节点也可以
# print(LLOCA_model.check_model())