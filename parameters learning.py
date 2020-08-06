#Step 1: Generate some data
# Use the alarm model to generate data from it.

from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling


alarm_model = get_example_model('alarm')
samples = BayesianModelSampling(alarm_model).forward_sample(size=int(1e5))
print(samples.head())

#Step 2: Define a model structure
# Defining the Bayesian Model structure

from pgmpy.models import BayesianModel

model_struct = BayesianModel(ebunch=alarm_model.edges())
print(model_struct.nodes())

#Step 3: Learning the model parameters
# Fitting the model using Maximum Likelihood Estimator

from pgmpy.estimators import MaximumLikelihoodEstimator

mle = MaximumLikelihoodEstimator(model=model_struct, data=samples)

# Estimating the CPD for a single node.
print(mle.estimate_cpd(node='FIO2'))
print(mle.estimate_cpd(node='CVP'))

# Estimating CPDs for all the nodes in the model
print(mle.get_parameters()[:10]) # Show just the first 10 CPDs in the output

# Verifying that the learned parameters are almost equal.
import numpy as np
print(np.allclose(alarm_model.get_cpds('FIO2').values, mle.estimate_cpd('FIO2').values, atol=0.01))

# Fitting the using Bayesian Estimator
from pgmpy.estimators import BayesianEstimator

best = BayesianEstimator(model=model_struct, data=samples)

print(best.estimate_cpd(node='FIO2', prior_type="BDeu", equivalent_sample_size=1000))
# Uniform pseudo count for each state. Can also accept an array of the size of CPD.
print(best.estimate_cpd(node='CVP', prior_type="dirichlet", pseudo_counts=100))

# Learning CPDs for all the nodes in the model. For learning all parameters with BDeU prior, a dict of
# pseudo_counts need to be provided
print(best.get_parameters(prior_type="BDeu", equivalent_sample_size=1000)[:10])

# Shortcut for learning all the parameters and adding the CPDs to the model.

model_struct = BayesianModel(ebunch=alarm_model.edges())
model_struct.fit(data=samples, estimator=MaximumLikelihoodEstimator)
print(model_struct.get_cpds('FIO2'))

model_struct = BayesianModel(ebunch=alarm_model.edges())
model_struct.fit(data=samples, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=1000)
print(model_struct.get_cpds('FIO2'))