import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from online_dual_estimation import build_filter
from utils import normal_log_prob


################################################################################
# Simple LDS with linear control

np.random.seed(0)
torch.manual_seed(0)

n_steps = 250
mc_samples = 1
num_gradient_steps = 10
true_stddev_emission = 0.1
true_stddev_transition = 1
true_A = 0.9
true_C = 1.0

use_kalman_filter = True

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

def transition_log_prob(x_t, x_prev, u_prev):
  # return normal_log_prob(x_t, x_prev + u_prev, true_stddev_transition)
  return normal_log_prob(
    x_t,
    learned_A * x_prev + u_prev,
    torch.exp(learned_log_stddev_transition)
  )

def emission_log_prob(y_t, x_t, u_t):
  # return normal_log_prob(y_t, x_t, true_stddev_emission)
  return normal_log_prob(
    y_t,
    learned_C * x_t,
    torch.exp(learned_log_stddev_emission)
  )

# net = torch.nn.Linear(4, 2)

# net = torch.nn.Sequential(
#   torch.nn.Linear(4, 25),
#   torch.nn.Tanh(),
#   torch.nn.Linear(25, 25),
#   torch.nn.Tanh(),
#   torch.nn.Linear(25, 2),
# )

# def kalman_filter(stuff):
#   # http://www.lce.hut.fi/~ssarkka/course_k2011/pdf/handout3.pdf
#   y_t = stuff[0]
#   u_prev = stuff[1]
#   mu_prev = stuff[2]
#   s_prev = stuff[3]
#
#   P_prev = torch.exp(2 * s_prev)
#   Q = true_stddev_transition ** 2
#   R = true_stddev_emission ** 2
#   H = 1
#
#   # Prediction
#   m_prime = true_A * mu_prev
#   P_prime = true_A * P_prev * true_A + Q
#
#   # Update
#   v = y_t - H * m_prime
#   S = H * P_prime * H + R
#   K = P_prime * H / S
#   m = m_prime + K * v
#   P = P_prime - K * S * K
#   return torch.cat([m, 0.5 * torch.log(P)])

def build_kalman_filter(log_stddev_transition, log_stddev_emission, A):
  def kalman_filter(stuff):
    # http://www.lce.hut.fi/~ssarkka/course_k2011/pdf/handout3.pdf
    y_t = stuff[0]
    u_prev = stuff[1]
    mu_prev = stuff[2]
    s_prev = stuff[3]

    P_prev = torch.exp(2 * s_prev)
    # Q = stddev_transition ** 2
    # R = stddev_emission ** 2
    Q = torch.exp(2 * log_stddev_transition)
    R = torch.exp(2 * log_stddev_emission)
    H = 1

    # Prediction
    m_prime = A * mu_prev
    P_prime = A * P_prev * A + Q

    # Update
    v = y_t - H * m_prime
    S = H * P_prime * H + R
    K = P_prime * H / S
    m = m_prime + K * v
    P = P_prime - K * S * K
    return torch.cat([m, 0.5 * torch.log(P)])
  return kalman_filter

# true_kalman_filter = build_kalman_filter(
#   Variable(
#     torch.log(torch.FloatTensor([true_stddev_transition])),
#     requires_grad=False
#   ),
#   Variable(
#     torch.log(torch.FloatTensor([true_stddev_emission])),
#     requires_grad=False
#   ),
#   true_A
# )
learned_kalman_filter = build_kalman_filter(
  learned_log_stddev_transition,
  learned_log_stddev_emission,
  learned_A
)

learning_rate = 1e-2
opt_variables = (
  # (list(net.parameters()) if not use_kalman_filter else []) +
  ([learned_log_stddev_emission] if learned_log_stddev_emission.requires_grad else []) +
  ([learned_log_stddev_transition] if learned_log_stddev_transition.requires_grad else []) +
  ([learned_A] if learned_A.requires_grad else []) +
  ([learned_C] if learned_C.requires_grad else [])
)
# optimizer = torch.optim.Adam(opt_variables, lr=learning_rate)
optimizer = torch.optim.SGD(opt_variables, lr=learning_rate)

# step = build_filter(1, emission_log_prob, transition_log_prob, net, optimizer)
step = build_filter(1,
  emission_log_prob,
  transition_log_prob,
  # (kalman_filter if use_kalman_filter else net),
  learned_kalman_filter,
  # true_kalman_filter,
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
  # if t >= 100:
  #   learned_log_stddev_emission.requires_grad = True
  #   learned_log_stddev_transition.requires_grad = False
  #   learned_A.requires_grad = False
  # else:
  #   learned_log_stddev_emission.requires_grad = False
  #   learned_log_stddev_transition.requires_grad = False
  #   learned_A.requires_grad = False

  x_prev = Variable(Xs[-1], requires_grad=False)
  # y_prev = Variable(Ys[-1], requires_grad=False)
  u_prev = Variable(Us[-1], requires_grad=False)

  mu_prev = Variable(Mus[-1], requires_grad=False)
  s_prev = Variable(Ss[-1], requires_grad=False)

  # No control for now
  u_t = Variable(torch.FloatTensor([0.0]), requires_grad=False)

  # Sample from the true model
  x_t = true_A * x_prev + u_prev + true_stddev_transition * np.random.randn()
  y_t = true_C * x_t + true_stddev_emission * np.random.randn()

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
import matplotlib.gridspec as gridspec

# Plot the filtering results
plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(6, 2)
gs.update(hspace=0, wspace=0.3)

plt.subplot(gs[:3, 0])
plt.plot(np.arange(n_steps), [t[0] for t in Xs[1:]])
plt.errorbar(
  np.arange(n_steps),
  [t[0] for t in Mus[1:]],
  xerr=0,
  yerr=[3 * np.exp(t[0]) for t in Ss[1:]]
)
plt.legend(['true x_t', 'posterior with 3 std. dev. error bars'])
plt.ylabel('latent space')

plt.subplot(gs[3:, 0])
plt.plot(np.arange(n_steps), [t[0] for t in Xs[1:]])
# plt.plot(np.arange(n_steps), [t[0] for t in Ys[1:]])
plt.legend(['observed y_t'])
# plt.legend(['x_t', 'observed y_t'])
plt.ylabel('observed space')

plt.suptitle('Dual estimation on a 1-d LDS model\n{} learning rate = {}, emission stddev = {}, transition stddev = {}, A = {}, C = {}\ninference = {}, mc_samples = {}, num_gradient_steps = {}'.format(optimizer.__class__.__name__, learning_rate, true_stddev_emission, true_stddev_transition, true_A, true_C, ('kalman filter' if use_kalman_filter else 'inference net'), mc_samples, num_gradient_steps))

# Plot the variational model params
plt.subplot(gs[:2, 1])
plt.plot(np.arange(n_steps), learned_log_stddev_transitions_per_iter[1:])
plt.axhline(np.log(true_stddev_transition), color='grey', linestyle='dotted')
if not learned_log_stddev_transition.requires_grad:
  plt.annotate('OFF', xy=(15, 15), xycoords='axes points', color='grey', size=24)
plt.legend(['estimated', 'ground truth'])
plt.ylabel('transition noise\nlog std. dev.')

plt.subplot(gs[2:4, 1])
plt.plot(np.arange(n_steps), learned_log_stddev_emissions_per_iter[1:])
plt.axhline(np.log(true_stddev_emission), color='grey', linestyle='dotted')
if not learned_log_stddev_emission.requires_grad:
  plt.annotate('OFF', xy=(15, 15), xycoords='axes points', color='grey', size=24)
plt.legend(['estimated', 'ground truth'])
plt.ylabel('emission noise\nlog std. dev.')

plt.subplot(gs[4:, 1])
plt.plot(np.arange(n_steps), learned_A_per_iter[1:])
plt.axhline(true_A, color='grey', linestyle='dotted')
if not learned_A.requires_grad:
  plt.annotate('OFF', xy=(15, 15), xycoords='axes points', color='grey', size=24)
plt.legend(['estimated', 'ground truth'])
plt.ylabel('A')

plt.subplots_adjust(top=0.85, left=0.1, right=0.95)

plt.show()
