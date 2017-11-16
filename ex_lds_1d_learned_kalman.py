"""Simple LDS in 1d with Kalman filtering on the learned parameters as the
inference "net."
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

from lds_1d_model import build_1d_lds_model, forward_sample, pretty_plot
from utils import build_1d_kalman_filter
from variational_dual_estimation import build_conditional_filter, run_conditional_filter


np.random.seed(123)
torch.manual_seed(123)

x_dim = 1
y_dim = 1
u_dim = 1

num_sequences = 1
num_steps = 2500
mc_samples = 1
# num_gradient_steps = 10
true_stddev_emission = 1
true_stddev_transition = 0.5
true_A = 0.97
true_C = 1.0

# The parameters of our model that will be fit along with the filtering process
# learned_precision_emission = Variable(torch.abs(torch.randn(1)), requires_grad=True)
# learned_precision_transition = Variable(torch.abs(torch.randn(1)), requires_grad=True)
# learned_A = Variable(torch.randn(1), requires_grad=True)
# learned_C = Variable(torch.randn(1), requires_grad=True)

learned_precision_emission = Variable(
  torch.FloatTensor([1.0 / (true_stddev_emission ** 2)]),
  requires_grad=False
)
learned_precision_transition = Variable(
  torch.FloatTensor([1.0 / (true_stddev_transition ** 2)]),
  requires_grad=False
)

# learned_precision_emission = Variable(
#   torch.log(torch.FloatTensor([true_stddev_emission])),
#   requires_grad=True
# )
# learned_precision_transition = Variable(
#   torch.log(torch.FloatTensor([true_stddev_transition])),
#   requires_grad=True
# )
# learned_A = Variable(torch.FloatTensor([true_A]), requires_grad=True)
learned_A = Variable(torch.FloatTensor([0.5]), requires_grad=True)
learned_C = Variable(torch.FloatTensor([true_C]), requires_grad=False)

emission_log_prob, transition_log_prob = build_1d_lds_model(
  learned_precision_emission,
  learned_precision_transition,
  learned_A,
  learned_C
)

learned_kalman_filter = build_1d_kalman_filter(
  torch.pow(learned_precision_emission, -1.0),
  torch.pow(learned_precision_transition, -1.0),
  learned_A,
  learned_C
)

learning_rate = 1e-3
opt_variables = (
  ([learned_precision_emission] if learned_precision_emission.requires_grad else []) +
  ([learned_precision_transition] if learned_precision_transition.requires_grad else []) +
  ([learned_A] if learned_A.requires_grad else []) +
  ([learned_C] if learned_C.requires_grad else [])
)
# optimizer = torch.optim.Adam(opt_variables, lr=learning_rate)
optimizer = torch.optim.SGD(opt_variables, lr=learning_rate)

step = build_conditional_filter(
  x_dim,
  emission_log_prob,
  transition_log_prob,
  learned_kalman_filter
)

Us = torch.zeros(num_sequences, num_steps, u_dim)

learned_log_stddev_emissions_per_iter = [learned_precision_emission.data[0]]
learned_precision_transitions_per_iter = [learned_precision_transition.data[0]]
learned_A_per_iter = [learned_A.data[0]]
loss_per_iter = []

def callback(info):
  print(info['t'])

  learned_log_stddev_emissions_per_iter.append(
    learned_precision_emission.data[0]
  )
  learned_precision_transitions_per_iter.append(
    learned_precision_transition.data[0]
  )
  learned_A_per_iter.append(learned_A.data[0])
  loss_per_iter.append(info['loss'].data[0])

Xs, Ys, Mus, Ss, callback_log = run_conditional_filter(
  lambda *args: step(*args, mc_samples=mc_samples),
  lambda x_prev, u_prev: forward_sample(
    true_stddev_emission,
    true_stddev_transition,
    true_A,
    true_C,
    x_prev,
    u_prev
  ),
  optimizer,
  num_steps,
  num_sequences,
  x_dim,
  y_dim,
  u_dim,
  control=Us,
  callback=callback
)

################################################################################
# Plotting

plt.plot(np.array(loss_per_iter) / num_sequences / mc_samples)
# plt.plot(np.array(loss_per_iter) / np.arange(1, len(loss_per_iter) + 1) / num_sequences / mc_samples)
plt.title('Variational loss per time step under the "bad" parameters')

for i in range(num_sequences):
  pretty_plot(
    num_steps,
    Xs[i],
    Ys[i],
    Us[i],
    Mus[i],
    Ss[i],
    learned_log_stddev_emissions_per_iter,
    learned_precision_transitions_per_iter,
    learned_A_per_iter,
    true_stddev_emission,
    true_stddev_transition,
    true_A,
    learned_precision_emission.requires_grad,
    learned_precision_transition.requires_grad,
    learned_A.requires_grad
  )
  plt.suptitle((
    'Dual estimation on a 1-d LDS model with Kalman filtering on the learned '
    'parameters\n'
    'optimizer = {}, learning rate = {}, emission stddev = {}, '
    'transition stddev = {}, A = {}, C = {}\n'
    'mc_samples = {}, num_sequences = {}'
  ).format(
    optimizer.__class__.__name__,
    learning_rate,
    true_stddev_emission,
    true_stddev_transition,
    true_A,
    true_C,
    mc_samples,
    num_sequences
  ))

  plt.show()
