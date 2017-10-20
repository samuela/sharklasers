import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from lds_1d_model import build_1d_lds_model, forward_sample, pretty_plot
from online_dual_estimation import build_filter
from utils import build_1d_kalman_filter


################################################################################
# Simple LDS in 1d with Kalman filtering on the true parameters as the inference
# "net."

np.random.seed(0)
torch.manual_seed(0)

n_steps = 250
mc_samples = 1
num_gradient_steps = 10
true_stddev_emission = 0.1
true_stddev_transition = 1
true_A = 0.9
true_C = 1.0

# The parameters of our model that will be fit along with the filtering process
learned_log_stddev_transition = Variable(torch.randn(1), requires_grad=True)
learned_log_stddev_emission = Variable(torch.randn(1), requires_grad=True)
learned_A = Variable(torch.randn(1), requires_grad=True)
# learned_C = Variable(torch.randn(1), requires_grad=True)

# learned_log_stddev_transition = Variable(
#   torch.log(torch.FloatTensor([true_stddev_transition])),
#   requires_grad=True
# )
# learned_log_stddev_emission = Variable(
#   torch.log(torch.FloatTensor([true_stddev_emission])),
#   requires_grad=True
# )
# learned_A = Variable(torch.FloatTensor([true_A]), requires_grad=True)
learned_C = Variable(torch.FloatTensor([true_C]), requires_grad=False)

emission_log_prob, transition_log_prob = build_1d_lds_model(
  learned_log_stddev_emission,
  learned_log_stddev_transition,
  learned_A,
  learned_C
)

true_kalman_filter = build_1d_kalman_filter(
  Variable(
    torch.log(torch.FloatTensor([true_stddev_emission])),
    requires_grad=False
  ),
  Variable(
    torch.log(torch.FloatTensor([true_stddev_transition])),
    requires_grad=False
  ),
  Variable(torch.FloatTensor([true_A]), requires_grad=False)
)

learning_rate = 1e-2
opt_variables = (
  ([learned_log_stddev_emission] if learned_log_stddev_emission.requires_grad else []) +
  ([learned_log_stddev_transition] if learned_log_stddev_transition.requires_grad else []) +
  ([learned_A] if learned_A.requires_grad else []) +
  ([learned_C] if learned_C.requires_grad else [])
)
# optimizer = torch.optim.Adam(opt_variables, lr=learning_rate)
optimizer = torch.optim.SGD(opt_variables, lr=learning_rate)

step = build_filter(
  1,
  emission_log_prob,
  transition_log_prob,
  true_kalman_filter,
  optimizer
)

Xs = [torch.FloatTensor([0.0])]
Ys = [torch.FloatTensor([0.0])]
Us = [torch.FloatTensor([0.0])]

Mus = [torch.FloatTensor([0.0])]
Ss = [torch.FloatTensor([0.0])]

learned_log_stddev_transitions_per_iter = [learned_log_stddev_transition.data[0]]
learned_log_stddev_emissions_per_iter = [learned_log_stddev_emission.data[0]]
learned_A_per_iter = [learned_A.data[0]]

for t in range(n_steps):
  print(t)
  x_prev = Variable(Xs[-1], requires_grad=False)
  u_prev = Variable(Us[-1], requires_grad=False)

  mu_prev = Variable(Mus[-1], requires_grad=False)
  s_prev = Variable(Ss[-1], requires_grad=False)

  # No control for now
  u_t = Variable(torch.FloatTensor([0.0]), requires_grad=False)

  # Sample from the true model
  x_t, y_t = forward_sample(
    true_stddev_emission,
    true_stddev_transition,
    true_A,
    true_C,
    x_prev,
    u_prev
  )

  mu_t, s_t = step(
    y_t,
    u_t,
    u_prev,
    mu_prev,
    s_prev,
    mc_samples=mc_samples,
    num_gradient_steps=num_gradient_steps
  )

  Xs.append(x_t.data.clone())
  Ys.append(y_t.data.clone())
  Us.append(u_t.data.clone())
  Mus.append(mu_t.data.clone())
  Ss.append(s_t.data.clone())
  learned_log_stddev_emissions_per_iter.append(learned_log_stddev_emission.data[0])
  learned_log_stddev_transitions_per_iter.append(learned_log_stddev_transition.data[0])
  learned_A_per_iter.append(learned_A.data[0])

################################################################################
# Plotting

import matplotlib.pyplot as plt

pretty_plot(
  n_steps,
  Xs,
  Ys,
  Us,
  Mus,
  Ss,
  learned_log_stddev_emissions_per_iter,
  learned_log_stddev_transitions_per_iter,
  learned_A_per_iter,
  true_stddev_emission,
  true_stddev_transition,
  true_A,
  learned_log_stddev_emission.requires_grad,
  learned_log_stddev_transition.requires_grad,
  learned_A.requires_grad
)
plt.suptitle((
  'Dual estimation on a 1-d LDS model with Kalman filtering on the true '
    'parameters\n'
  'optimizer = {}, learning rate = {}, emission stddev = {}, '
    'transition stddev = {}, A = {}, C = {}\n'
  'mc_samples = {}, num_gradient_steps = {}'
).format(
  optimizer.__class__.__name__,
  learning_rate,
  true_stddev_emission,
  true_stddev_transition,
  true_A,
  true_C,
  mc_samples,
  num_gradient_steps
))
plt.show()
