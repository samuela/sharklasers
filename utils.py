import numpy as np
import torch

def normal_log_prob(x, mu, sigma):
  return -0.5 * torch.log(2.0 * np.pi * sigma * sigma) - 0.5 * (((x - mu) / sigma) ** 2.0)
  # return -0.5 * torch.log(2.0 * np.pi * sigma * sigma) - 0.5 * torch.pow((x - mu) / sigma, 2.0)

def build_1d_kalman_filter(log_stddev_emission, log_stddev_transition, A):
  """
  Construct a Kalman filter.

  Parameters
  ==========
  log_stddev_emission : torch.autograd.Variable
  log_stddev_transition : torch.autograd.Variable
  A : torch.autograd.Variable
  """
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
