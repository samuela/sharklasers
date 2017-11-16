import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from variational_dual_estimation import *
from utils import normal_log_prob

np.random.seed(0)
torch.manual_seed(0)

mc_samples = 1
num_gradient_steps = 1

x_dim = 2
u_dim = 1
y_dim = 200

tau_r = 1
tau_theta = 1
step_size = 0.1

true_model = {
  # The resting state radius
  'r_0': 1,

  # Gaussian noise on the latent state
  'x_noise_stddev': 0.005,

  'C': Variable(torch.randn(y_dim, 2), requires_grad=False),
  'b': Variable(-5 + torch.randn(y_dim), requires_grad=False)
}

num_rbfs = 20
estimated_model = {
  # 'C': Variable(torch.randn(y_dim, x_dim), requires_grad=True),
  # 'b': Variable(torch.randn(y_dim), requires_grad=True),
  'C': Variable(true_model['C'].data, requires_grad=False),
  'b': Variable(true_model['b'].data, requires_grad=False),
  'rbf_centers': Variable(2.5 * torch.randn(num_rbfs, x_dim), requires_grad=True),
  'rbf_gammas': Variable(torch.ones(num_rbfs), requires_grad=True),
  'W_g': Variable(torch.zeros(x_dim, num_rbfs), requires_grad=True),
  'W_B': Variable(torch.zeros(x_dim * u_dim, num_rbfs), requires_grad=True),
  'x_noise_stddev': Variable(torch.FloatTensor([true_model['x_noise_stddev']]), requires_grad=False)
}

# `u_prev` here is the same as `I` in the paper
def true_dynamics_forward_sample(x_prev, u_prev):
  r = torch.norm(x_prev)
  theta = torch.atan2(x_prev[1:], x_prev[:1])
  delta_r = (true_model['r_0'] - r) / tau_r
  delta_theta = u_prev / tau_theta
  new_r = r + step_size * delta_r
  new_theta = theta + step_size * delta_theta
  return (
    new_r * torch.cat([torch.cos(new_theta), torch.sin(new_theta)])
  + true_model['x_noise_stddev'] * Variable(torch.randn(x_dim), requires_grad=False)
  )

def transition_log_prob(x_t, x_prev, u_prev):
  phi = torch.exp(
    -0.5
  * estimated_model['rbf_gammas']
  * torch.sum(torch.pow(x_prev - estimated_model['rbf_centers'], 2), 1)
  )
  B = (estimated_model['W_B'] @ phi).view(x_dim, u_dim)
  center = x_prev + estimated_model['W_g'] @ phi + B @ u_prev
  return sum([
    normal_log_prob(x_t[i], center[i], estimated_model['x_noise_stddev'])
    for i in range(x_dim)
  ])

def emission_log_prob(y_t, x_t, u_t):
  logits = estimated_model['C'] @ x_t + estimated_model['b']
  return F.binary_cross_entropy_with_logits(logits, y_t)

mlp_hidden_units = 100
net = torch.nn.Sequential(
  torch.nn.Linear(y_dim + u_dim + 2 * x_dim, mlp_hidden_units),
  torch.nn.Sigmoid(),
  # torch.nn.Tanh(),
  torch.nn.Linear(mlp_hidden_units, 2 * x_dim),
)

learning_rate = 1e-3
opt_variables = (
  list(net.parameters()) +
  [var for var in estimated_model.values() if var.requires_grad]
)
optimizer = torch.optim.Adam(opt_variables, lr=learning_rate)
# optimizer = torch.optim.SGD(opt_variables, lr=learning_rate)

step = build_filter(x_dim, emission_log_prob, transition_log_prob, net, optimizer)

Xs = [torch.FloatTensor([0.0, 0.0])]
Ys = []
Us = [torch.FloatTensor([1.0])]

Mus = [torch.FloatTensor([1.0, 0.0])]
Ss = [torch.FloatTensor([0.0, 0.0])]

models = [{k: v.data.clone() for k, v in estimated_model.items()}]

for t in range(1000):
  print(t)
  x_prev = Variable(Xs[-1], requires_grad=False)
  u_prev = Variable(Us[-1], requires_grad=False)

  mu_prev = Variable(Mus[-1], requires_grad=False)
  s_prev = Variable(Ss[-1], requires_grad=False)

  # Simulate the true dynamics
  x_t = true_dynamics_forward_sample(x_prev, u_prev)
  y_t = torch.bernoulli(torch.sigmoid(true_model['C'] @ x_t + true_model['b']))

  # Control: +1 for counter-clockwise and -1 for clockwise
  u_t = Variable(torch.FloatTensor([2.0]), requires_grad=False)

  mu_t, s_t = step(
    y_t,
    u_t,
    u_prev,
    mu_prev,
    s_prev,
    mc_samples=mc_samples,
    num_gradient_steps=num_gradient_steps
  )

  # print(estimated_model['rbf_centers'].data)
  # print(estimated_model['rbf_gammas'].data)

  Xs.append(x_t.data.clone())
  Ys.append(y_t.data.clone())
  Us.append(u_t.data.clone())
  Mus.append(mu_t.data.clone())
  Ss.append(s_t.data.clone())
  models.append({k: v.data.clone() for k, v in estimated_model.items()})

################################################################################
for k, v in estimated_model.items():
  if not np.all(np.isfinite(v.data.numpy())):
    print('NaNs in {}!'.format(k))

import matplotlib.pyplot as plt

plt.figure()
plt.plot(
  [x[0] for x in Xs[1:]],
  [x[1] for x in Xs[1:]],
  marker='.',
  alpha=0.5
)
plt.plot(
  [x[0] for x in Mus[1:]],
  [x[1] for x in Mus[1:]],
  marker='.',
  alpha=0.5,
  color='r'
)
# plt.scatter(
#   [estimated_model['rbf_centers'].data[i, 0] for i in range(num_rbfs)],
#   [estimated_model['rbf_centers'].data[i, 1] for i in range(num_rbfs)],
#   marker='x',
#   color='k'
# )
plt.quiver(
  [estimated_model['rbf_centers'].data[i, 0] for i in range(num_rbfs)],
  [estimated_model['rbf_centers'].data[i, 1] for i in range(num_rbfs)],
  [estimated_model['W_g'].data[0, i] for i in range(num_rbfs)],
  [estimated_model['W_g'].data[1, i] for i in range(num_rbfs)],
)
plt.legend(['true', 'filtered', 'RBF centers'])
plt.axis('equal')
# plt.axis([-1.05, 1.05, -1.05, 1.05])

plt.matshow(np.array([y.numpy() for y in Ys]).T)

plt.show()
