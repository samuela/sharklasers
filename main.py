import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn


def agglomerate(ts):
  return torch.cat([t.view(-1) for t in ts])

def mvn_entropy(s):
  """
  Calculate the entropy of a multivariate normal distribution given the log
  standard deviations up to an additive constant (which it turns out is
  `N / 2 * log(2 * pi * e)`).

  Parameters
  ==========
  s : torch.Tensor
    The log standard deviations.
  """
  return torch.sum(s)

def build_filter(N, emission_log_prob, transition_log_prob, net, optimizer):
  """
  Construct a online dual estimation filter as defined by Algorithm 1 in
  https://arxiv.org/abs/1707.09049. `s_t` is not quite the same thing because
  somehow they're taking a sqrt of theirs which doesn't make much sense in
  general. In this case it denotes the log standard deviation.

  Parameters
  ==========
  N : int
    The dimension of the latent space, ie. the dimension of x_t.
  emission_log_prob : function
  transition_log_prob : function
  net : torch something
  optimizer : torch.optim something
  """
  def filter_step(y_t, u_prev, mu_prev, s_prev, mc_samples=1):
    # eps_t = Variable(torch.randn(N), requires_grad=False)
    # eps_prev = Variable(torch.randn(N), requires_grad=False)

    net_out = net(agglomerate([y_t, u_prev, mu_prev, s_prev]))
    mu_t, s_t = torch.split(net_out, N)

    # x_squiggle_t = mu_t + torch.exp(s_t) * eps_t
    # x_squiggle_prev = mu_prev + torch.exp(s_prev) * eps_prev

    # loss = -1 * (
    #   emission_log_prob(y_t, x_squiggle_t, u_t)
    # + transition_log_prob(x_squiggle_t, x_squiggle_prev, u_prev)
    # + mvn_entropy(s_t)
    # )

    # Instead of dividing the sums by mc_samples, we can just multiply this term
    # by mc_samples.
    loss = -mc_samples * mvn_entropy(s_t)
    # print('-mc_samples * mvn_entropy(s_t)', (-mc_samples * mvn_entropy(s_t)).data[0])
    for _ in range(mc_samples):
      eps_t = Variable(torch.randn(N), requires_grad=False)
      eps_prev = Variable(torch.randn(N), requires_grad=False)

      x_squiggle_t = mu_t + torch.exp(s_t) * eps_t
      x_squiggle_prev = mu_prev + torch.exp(s_prev) * eps_prev

      print('emission_log_prob(y_t, x_squiggle_t, u_t)', emission_log_prob(y_t, x_squiggle_t, u_t).data[0])
      print('transition_log_prob(x_squiggle_t, x_squiggle_prev, u_prev)', transition_log_prob(x_squiggle_t, x_squiggle_prev, u_prev).data[0])
      loss -= (
        emission_log_prob(y_t, x_squiggle_t, u_t)
      + transition_log_prob(x_squiggle_t, x_squiggle_prev, u_prev)
      )

    optimizer.zero_grad()
    loss.backward()
    print('variational_log_stddev_emission.data', variational_log_stddev_emission.data[0])
    print('variational_log_stddev_emission.grad.data', variational_log_stddev_emission.grad.data[0])
    optimizer.step()

    return mu_t, s_t

  return filter_step

################################################################################
# Simple LDS with linear control

np.random.seed(0)
torch.manual_seed(0)

n_steps = 250
mc_samples = 25
true_stddev_emission = 1
true_stddev_transition = 1
true_A = 0.9

def normal_log_prob(x, mu, sigma):
  return -0.5 * (((x - mu) / sigma) ** 2)

# The parameters of our model that will be fit along with the filtering process
variational_log_stddev_transition = Variable(torch.randn(1), requires_grad=True)
variational_log_stddev_emission = Variable(torch.randn(1), requires_grad=True)
variational_A = Variable(torch.randn(1), requires_grad=True)

# variational_log_stddev_transition = Variable(
#   torch.log(torch.FloatTensor([true_stddev_transition])),
#   requires_grad=False
# )
# variational_log_stddev_emission = Variable(
#   torch.log(torch.FloatTensor([true_stddev_emission])),
#   requires_grad=False
# )
# variational_A = Variable(torch.FloatTensor([true_A]), requires_grad=False)
# variational_A = Variable(torch.FloatTensor([-10]), requires_grad=True)

def trans_lp(x_t, x_prev, u_prev):
  # return normal_log_prob(x_t, x_prev + u_prev, true_stddev_transition)
  return normal_log_prob(
    x_t,
    variational_A * x_prev + u_prev,
    torch.exp(variational_log_stddev_transition)
  )

def emiss_lp(y_t, x_t, u_t):
  # return normal_log_prob(y_t, x_t, true_stddev_emission)
  return normal_log_prob(y_t, x_t, torch.exp(variational_log_stddev_emission))

# net = torch.nn.Linear(4, 2)

def kalman_filter(stuff):
  # http://www.lce.hut.fi/~ssarkka/course_k2011/pdf/handout3.pdf
  y_t = stuff[0]
  u_prev = stuff[1]
  mu_prev = stuff[2]
  s_prev = stuff[3]

  P_prev = torch.exp(2 * s_prev)
  Q = true_stddev_transition ** 2
  R = true_stddev_emission ** 2
  H = 1

  # Prediction
  m_prime = true_A * mu_prev
  P_prime = true_A * P_prev * true_A + Q

  # Update
  v = y_t - H * m_prime
  S = H * P_prime * H + R
  K = P_prime * H / S
  m = m_prime + K * v
  P = P_prime - K * S * K
  return torch.cat([m, 0.5 * torch.log(P)])

learning_rate = 1e-3
opt_variables = (
  # list(net.parameters()) +
  ([variational_log_stddev_emission] if variational_log_stddev_emission.requires_grad else []) +
  ([variational_log_stddev_transition] if variational_log_stddev_transition.requires_grad else []) +
  ([variational_A] if variational_A.requires_grad else [])
)
# optimizer = torch.optim.Adam(opt_variables, lr=learning_rate)
optimizer = torch.optim.SGD(opt_variables, lr=learning_rate)

# step = build_filter(1, emiss_lp, trans_lp, net, optimizer)
step = build_filter(1, emiss_lp, trans_lp, kalman_filter, optimizer)

Xs = [torch.FloatTensor([0.0])]
Ys = [torch.FloatTensor([0.0])]
Us = [torch.FloatTensor([0.0])]

Mus = [torch.FloatTensor([0.0])]
Ss = [torch.FloatTensor([0.0])]

variational_log_stddev_transitions_per_iter = [variational_log_stddev_transition.data[0]]
variational_log_stddev_emissions_per_iter = [variational_log_stddev_emission.data[0]]
variational_A_per_iter = [variational_A.data[0]]

for t in range(n_steps):
  print(t)
  x_prev = Variable(Xs[-1], requires_grad=False)
  y_prev = Variable(Ys[-1], requires_grad=False)
  u_prev = Variable(Us[-1], requires_grad=False)

  mu_prev = Variable(Mus[-1], requires_grad=False)
  s_prev = Variable(Ss[-1], requires_grad=False)

  # No control for now
  u_t = Variable(torch.FloatTensor([0.0]), requires_grad=False)

  # Sample from the true model
  x_t = true_A * x_prev + u_prev + true_stddev_transition * np.random.randn()
  y_t = x_t + true_stddev_emission * np.random.randn()

  mu_t, s_t = step(y_t, u_prev, mu_prev, s_prev, mc_samples=mc_samples)

  Xs.append(x_t.data)
  Ys.append(y_t.data)
  Us.append(u_t.data)
  Mus.append(mu_t.data)
  Ss.append(s_t.data)
  variational_log_stddev_emissions_per_iter.append(variational_log_stddev_emission.data[0])
  variational_log_stddev_transitions_per_iter.append(variational_log_stddev_transition.data[0])
  variational_A_per_iter.append(variational_A.data[0])

  print()

################################################################################
# Plotting

plot_log = True

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
plt.xlabel('latent space')

plt.subplot(gs[3:, 0])
plt.plot(np.arange(n_steps), [t[0] for t in Ys[1:]])
plt.legend(['observed y_t'])
plt.ylabel('observed space')

plt.suptitle('Dual estimation on a 1-d LDS model\n{} learning rate = {}, emission stddev = {}, transition stddev = {}, A = {}, mc_samples = {}'.format(optimizer.__class__.__name__, learning_rate, true_stddev_emission, true_stddev_transition, true_A, mc_samples))

# Plot the variational model params
plt.subplot(gs[:2, 1])
if plot_log:
  plt.plot(np.arange(n_steps), variational_log_stddev_transitions_per_iter[1:])
  plt.axhline(np.log(true_stddev_transition), color='grey', linestyle='dotted')
  if not variational_log_stddev_transition.requires_grad:
    plt.annotate('OFF', xy=(15, 15), xycoords='axes points', color='grey', size=24)
  plt.legend(['variational estimate', 'ground truth'])
  plt.ylabel('transition noise\nlog std. dev.')
else:
  plt.plot(np.arange(n_steps), np.exp(variational_log_stddev_transitions_per_iter)[1:])
  plt.axhline(true_stddev_transition, color='grey', linestyle='dotted')
  if not variational_log_stddev_transition.requires_grad:
    plt.annotate('OFF', xy=(15, 15), xycoords='axes points', color='grey', size=24)
  plt.legend(['variational estimate', 'ground truth'])
  plt.ylabel('transition noise\nstd. dev.')

plt.subplot(gs[2:4, 1])
if plot_log:
  plt.plot(np.arange(n_steps), variational_log_stddev_emissions_per_iter[1:])
  plt.axhline(np.log(true_stddev_emission), color='grey', linestyle='dotted')
  if not variational_log_stddev_emission.requires_grad:
    plt.annotate('OFF', xy=(15, 15), xycoords='axes points', color='grey', size=24)
  plt.legend(['variational estimate', 'ground truth'])
  plt.ylabel('emission noise\nlog std. dev.')
else:
  plt.plot(np.arange(n_steps), np.exp(variational_log_stddev_emissions_per_iter)[1:])
  plt.axhline(true_stddev_emission, color='grey', linestyle='dotted')
  if not variational_log_stddev_emission.requires_grad:
    plt.annotate('OFF', xy=(15, 15), xycoords='axes points', color='grey', size=24)
  plt.legend(['variational estimate', 'ground truth'])
  plt.ylabel('emission noise\nstd. dev.')

plt.subplot(gs[4:, 1])
plt.plot(np.arange(n_steps), variational_A_per_iter[1:])
plt.axhline(true_A, color='grey', linestyle='dotted')
if not variational_A.requires_grad:
  plt.annotate('OFF', xy=(15, 15), xycoords='axes points', color='grey', size=24)
plt.legend(['variational estimate', 'ground truth'])
plt.ylabel('A')

# plt.suptitle('Variational model parameters per iteration\nlearning rate = {}, emission stddev = {}, transition stddev = {}, A = {}'.format(learning_rate, true_stddev_emission, true_stddev_transition, true_A))

plt.show()
