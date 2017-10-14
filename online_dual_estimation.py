import torch
from torch.autograd import Variable


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
  def filter_step(
    y_t,
    u_t,
    u_prev,
    mu_prev,
    s_prev,
    mc_samples=1,
    num_gradient_steps=1
  ):
    for _ in range(num_gradient_steps):
      net_out = net(agglomerate([y_t, u_prev, mu_prev, s_prev]))
      mu_t, s_t = torch.split(net_out, N)

      # Instead of dividing the sums by mc_samples, we can just multiply this term
      # by mc_samples.
      loss = -mc_samples * mvn_entropy(s_t)
      for _ in range(mc_samples):
        eps_t = Variable(torch.randn(N), requires_grad=False)
        eps_prev = Variable(torch.randn(N), requires_grad=False)

        x_squiggle_t = mu_t + torch.exp(s_t) * eps_t
        x_squiggle_prev = mu_prev + torch.exp(s_prev) * eps_prev

        loss -= (
          emission_log_prob(y_t, x_squiggle_t, u_t)
        + transition_log_prob(x_squiggle_t, x_squiggle_prev, u_prev)
        )

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    return mu_t, s_t

  return filter_step
