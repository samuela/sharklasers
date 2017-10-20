import numpy as np

import torch

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import normal_log_prob


def build_1d_lds_model(log_stddev_emission, log_stddev_transition, A, C):
  """
  Construct model potentials for a 1d LDS model.

  Parameters
  ==========
  log_stddev_emission : torch.autograd.Variable
  log_stddev_transition : torch.autograd.Variable
  A : torch.autograd.Variable
  C : torch.autograd.Variable

  Returns
  =======
  A pair (emission_log_prob, transition_log_prob) with the two model potentials.
  """
  def emission_log_prob(y_t, x_t, u_t):
    # return normal_log_prob(y_t, x_t, true_stddev_emission)
    return normal_log_prob(
      y_t,
      C * x_t,
      torch.exp(log_stddev_emission)
    )

  def transition_log_prob(x_t, x_prev, u_prev):
    # return normal_log_prob(x_t, x_prev + u_prev, true_stddev_transition)
    return normal_log_prob(
      x_t,
      A * x_prev + u_prev,
      torch.exp(log_stddev_transition)
    )

  return emission_log_prob, transition_log_prob

def forward_sample(stddev_emission, stddev_transition, A, C, x_prev, u_prev):
  x_t = A * x_prev + u_prev + stddev_transition * np.random.randn()
  y_t = C * x_t + stddev_emission * np.random.randn()
  return x_t, y_t

def pretty_plot(
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
  learned_log_stddev_emission_on,
  learned_log_stddev_transition_on,
  learned_A_on
):
  # Plot the filtering results
  fig = plt.figure(figsize=(14, 6))
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

  # Plot the variational model params
  plt.subplot(gs[:2, 1])
  plt.plot(np.arange(n_steps), learned_log_stddev_transitions_per_iter[1:])
  plt.axhline(np.log(true_stddev_transition), color='grey', linestyle='dotted')
  if not learned_log_stddev_transition_on:
    plt.annotate('OFF', xy=(15, 15), xycoords='axes points', color='grey', size=24)
  plt.legend(['estimated', 'ground truth'])
  plt.ylabel('transition noise\nlog std. dev.')

  plt.subplot(gs[2:4, 1])
  plt.plot(np.arange(n_steps), learned_log_stddev_emissions_per_iter[1:])
  plt.axhline(np.log(true_stddev_emission), color='grey', linestyle='dotted')
  if not learned_log_stddev_emission_on:
    plt.annotate('OFF', xy=(15, 15), xycoords='axes points', color='grey', size=24)
  plt.legend(['estimated', 'ground truth'])
  plt.ylabel('emission noise\nlog std. dev.')

  plt.subplot(gs[4:, 1])
  plt.plot(np.arange(n_steps), learned_A_per_iter[1:])
  plt.axhline(true_A, color='grey', linestyle='dotted')
  if not learned_A_on:
    plt.annotate('OFF', xy=(15, 15), xycoords='axes points', color='grey', size=24)
  plt.legend(['estimated', 'ground truth'])
  plt.ylabel('A')

  plt.subplots_adjust(top=0.85, left=0.1, right=0.95)

  return fig
