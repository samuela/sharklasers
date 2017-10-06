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
  def filter_step(y_t, u_prev, mu_prev, s_prev):
    eps_t = Variable(torch.randn(N), requires_grad=False)
    eps_prev = Variable(torch.randn(N), requires_grad=False)

    net_out = net(agglomerate([y_t, u_prev, mu_prev, s_prev]))
    mu_t, s_t = torch.split(net_out, N)

    x_squiggle_t = mu_t + torch.exp(s_t) * eps_t
    x_squiggle_prev = mu_prev + torch.exp(s_prev) * eps_prev

    loss = -1 * (
      emission_log_prob(y_t, x_squiggle_t, u_t)
    + transition_log_prob(x_squiggle_t, x_squiggle_prev, u_prev)
    + mvn_entropy(s_t)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return mu_t, s_t

  return filter_step

################################################################################
# Simple LDS with linear control

np.random.seed(0)
torch.manual_seed(0)

n_steps = 1000
true_stddev_emission = 1
true_stddev_transition = 3
true_A = 0.7

def normal_log_prob(x, mu, sigma):
  return -0.5 * (((x - mu) / sigma) ** 2)

# The parameters of our model that will be fit along with the filtering process
# variational_log_stddev_transition = Variable(torch.randn(1), requires_grad=True)
# variational_log_stddev_emission = Variable(torch.randn(1), requires_grad=True)
variational_A = Variable(torch.randn(1), requires_grad=True)

variational_log_stddev_transition = Variable(torch.log(torch.FloatTensor([true_stddev_transition])), requires_grad=True)
variational_log_stddev_emission = Variable(torch.log(torch.FloatTensor([true_stddev_emission])), requires_grad=True)
# variational_A = Variable(torch.FloatTensor([true_A]), requires_grad=True)

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

net = torch.nn.Linear(4, 2)

learning_rate = 1e-5
# optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(
  list(net.parameters()) + [
    # variational_log_stddev_emission,
    # variational_log_stddev_transition,
    variational_A
  ],
  lr=learning_rate
)

step = build_filter(1, emiss_lp, trans_lp, net, optimizer)

Xs = [torch.FloatTensor([0.0])]
Ys = [torch.FloatTensor([0.0])]
Us = [torch.FloatTensor([0.0])]

Mus = [torch.FloatTensor([0.0])]
Ss = [torch.FloatTensor([0.0])]

variational_log_stddev_transitions_per_iter = [variational_log_stddev_transition.data[0]]
variational_log_stddev_emissions_per_iter = [variational_log_stddev_emission.data[0]]
variational_A_per_iter = [variational_A.data[0]]

for t in range(n_steps):
  # print(t)
  x_prev = Variable(Xs[-1], requires_grad=False)
  y_prev = Variable(Ys[-1], requires_grad=False)
  u_prev = Variable(Us[-1], requires_grad=False)

  mu_prev = Variable(Mus[-1], requires_grad=False)
  s_prev = Variable(Ss[-1], requires_grad=False)

  # No control for now
  u_t = Variable(torch.FloatTensor([0.0]), requires_grad=False)

  x_t = x_prev + u_prev + true_stddev_transition * np.random.randn()
  y_t = x_t + true_stddev_emission * np.random.randn()

  mu_t, s_t = step(y_t, u_prev, mu_prev, s_prev)

  Xs.append(x_t.data)
  Ys.append(y_t.data)
  Us.append(u_t.data)
  Mus.append(mu_t.data)
  Ss.append(s_t.data)
  variational_log_stddev_emissions_per_iter.append(variational_log_stddev_emission.data[0])
  variational_log_stddev_transitions_per_iter.append(variational_log_stddev_transition.data[0])
  variational_A_per_iter.append(variational_A.data[0])

print(net.weight)
print(net.bias)

import matplotlib.pyplot as plt

# Plot the filtering results
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(np.arange(n_steps + 1), [t[0] for t in Xs])
plt.errorbar(
  np.arange(n_steps + 1),
  [t[0] for t in Mus],
  xerr=0,
  yerr=[3 * np.exp(t[0]) for t in Ss]
)
plt.legend(['true x_t', 'posterior with 3 std. dev. error bars'])
plt.title('latent space')

plt.subplot(2, 1, 2)
plt.plot(np.arange(n_steps + 1), [t[0] for t in Ys])
plt.legend(['observed y_t'])
plt.title('observed space')

plt.suptitle('Dual estimation on a 1-d LDS model\nlearning rate = {}, emission stddev = {}, transition stddev = {}, A = {}'.format(learning_rate, true_stddev_emission, true_stddev_transition, true_A))

# Plot the variational model params
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(np.arange(n_steps + 1), variational_log_stddev_transitions_per_iter)
plt.axhline(np.log(true_stddev_transition), color='grey', linestyle='dotted')
plt.legend(['variational estimate', 'ground truth'])
plt.ylabel('transition noise\nlog std. dev.')

plt.subplot(3, 1, 2)
plt.plot(np.arange(n_steps + 1), variational_log_stddev_emissions_per_iter)
plt.axhline(np.log(true_stddev_emission), color='grey', linestyle='dotted')
plt.legend(['variational estimate', 'ground truth'])
plt.ylabel('emission noise\nlog std. dev.')

plt.subplot(3, 1, 3)
plt.plot(np.arange(n_steps + 1), variational_A_per_iter)
plt.axhline(true_A, color='grey', linestyle='dotted')
plt.legend(['variational estimate', 'ground truth'])
plt.ylabel('A')

plt.suptitle('Variational model parameters per iteration\nlearning rate = {}, emission stddev = {}, transition stddev = {}, A = {}'.format(learning_rate, true_stddev_emission, true_stddev_transition, true_A))

plt.show()
