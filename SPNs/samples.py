from spn.structure.leaves.parametric.Parametric import Categorical
from spn.structure.Base import Sum, Product
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up, eval_spn_bottom_up
from spn.io.Graphics import plot_spn
from spn.io.Text import spn_to_str_equation
from spn.algorithms.Inference import log_likelihood, likelihood
from spn.io.Symbolic import spn_to_sympy
from spn.gpu.TensorFlow import eval_tf

import numpy as np


class SPN:
  def __init__(self):
    p0 = Product(children=[Categorical(p=[0.3, 0.7], scope=1), Categorical(p=[0.4, 0.6], scope=2)])
    p1 = Product(children=[Categorical(p=[0.5, 0.5], scope=1), Categorical(p=[0.6, 0.4], scope=2)])
    s1 = Sum(weights=[0.3, 0.7], children=[p0, p1])
    p2 = Product(children=[Categorical(p=[0.2, 0.8], scope=0), s1])
    p3 = Product(children=[Categorical(p=[0.2, 0.8], scope=0), Categorical(p=[0.3, 0.7], scope=1)])
    p4 = Product(children=[p3, Categorical(p=[0.4, 0.6], scope=2)])

    self.spn = Sum(weights=[0.4, 0.6], children=[p2, p4])

    assign_ids(self.spn)
    rebuild_scopes_bottom_up(self.spn)

  def plot(self, name='images/spn.png'):
    plot_spn(self.spn, name)

  def to_text(self):
    return spn_to_str_equation(self.spn)

  def evaluate(self, inputs):
    return likelihood(self.spn, inputs)


spn = SPN()
# spn.plot()
# print(spn.to_text())


input = np.array([1.0, 0.0, 1.0]).reshape(-1, 3)
value = spn.evaluate(input)
print(value)
