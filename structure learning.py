
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models.BayesianModel import BayesianModel

# construct the tree graph structure
model = BayesianModel([('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E'), ('C', 'F')])
nx.draw_circular(model,  with_labels=True, arrowsize=30, node_size=800, alpha=0.3, font_weight='bold')
plt.savefig('fig1.png', bbox_inches='tight')

from pgmpy.factors.discrete import TabularCPD

# add CPD to each edge
cpd_a = TabularCPD('A', 2, [[0.4], [0.6]])
cpd_b = TabularCPD('B', 3, [[0.6,0.2],[0.3,0.5],[0.1,0.3]], evidence=['A'], evidence_card=[2])
cpd_c = TabularCPD('C', 2, [[0.3,0.4],[0.7,0.6]], evidence=['A'], evidence_card=[2])
cpd_d = TabularCPD('D', 3, [[0.5,0.3,0.1],[0.4,0.4,0.8],[0.1,0.3,0.1]], evidence=['B'], evidence_card=[3])
cpd_e = TabularCPD('E', 2, [[0.3,0.5,0.2],[0.7,0.5,0.8]], evidence=['B'], evidence_card=[3])
cpd_f = TabularCPD('F', 3, [[0.3,0.6],[0.5,0.2],[0.2,0.2]], evidence=['C'], evidence_card=[2])
model.add_cpds(cpd_a, cpd_b, cpd_c, cpd_d, cpd_e, cpd_f)

from pgmpy.sampling import BayesianModelSampling

# sample data from BN
inference = BayesianModelSampling(model)
df_data = inference.forward_sample(size=10000, return_type='dataframe')
print(df_data)

from pgmpy.estimators import TreeSearch

# learn graph structure
est = TreeSearch(df_data, root_node="A")
dag = est.estimate(estimator_type="chow-liu")
nx.draw_circular(dag, with_labels=True, arrowsize=30, node_size=800, alpha=0.3, font_weight='bold')
plt.savefig('fig2.png', bbox_inches='tight')
plt.show()

from pgmpy.estimators import BayesianEstimator

# there are many choices of parametrization, here is one example
model = BayesianModel(dag.edges())
model.fit(df_data, estimator=BayesianEstimator, prior_type='dirichlet', pseudo_counts=0.1)
print(model.get_cpds())