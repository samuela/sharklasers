import numpy as np

import torch
from torch.autograd import Variable


def normal_log_prob(x, mu, sigma):
  # return -0.5 * torch.log(2.0 * np.pi * sigma * sigma) - 0.5 * (((x - mu) / sigma) ** 2.0)
  return -0.5 * torch.log(2.0 * np.pi * sigma * sigma) - 0.5 * torch.pow((x - mu) / sigma, 2.0)
  # return torch.log(torch.rsqrt(2.0 * np.pi * torch.pow(sigma, 2)) * torch.exp(-1 * torch.pow(x - mu, 2) / (2 * torch.pow(sigma, 2))))

def normal_log_prob_precision(x, mu, tau):
  # return 0.5 * (-np.log(2 * np.pi) + torch.log(tau) - tau * torch.pow(x - mu, 2.0))
  return 0.5 * (torch.log(tau) - tau * torch.pow(x - mu, 2.0))

def build_1d_kalman_filter(variance_emission, variance_transition, A, C):
  """Construct a Kalman filter.

  Parameters
  ==========
  variance_emission : torch.autograd.Variable
  variance_transition : torch.autograd.Variable
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
    Q = variance_transition
    R = variance_emission
    # Q = torch.exp(2 * log_stddev_transition)
    # R = torch.exp(2 * log_stddev_emission)
    H = C

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

def nans_tensor(*args, **kwargs):
  return np.nan * torch.zeros(*args, **kwargs)

# https://github.com/pytorch/pytorch/issues/3223
def size_splits(tensor, split_sizes, dim=0):
  """Splits the tensor according to chunks of split_sizes.

  Parameters
  ==========
  tensor : Tensor
    The tensor to split.
  split_sizes : list(int)
    Sizes of chunks.
  dim : int
    Dimension along which to split the tensor.
  """
  if dim < 0:
    dim += tensor.dim()

  dim_size = tensor.size(dim)
  if dim_size != torch.sum(torch.Tensor(split_sizes)):
    raise KeyError("Sum of split sizes does not equal Tensor dim")

  splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

  return tuple(tensor.narrow(int(dim), int(start), int(length))
               for start, length in zip(splits, split_sizes))

def re_triu(vec, d):
  """Take a flattened upper triangular matrix and bring it back to life."""
  a = torch.zeros(d, d)
  a[torch.triu(torch.ones(d, d)) == 1] = vec
  return a

# https://github.com/pytorch/pytorch/issues/3423#issuecomment-343452452
# class Det(torch.autograd.Function):
#   """Matrix determinant. Input should be a PSD matrix."""

#   @staticmethod
#   def forward(ctx, x):
#     # output = torch.potrf(x).diag().prod(dim=0) ** 2
#     output =
#     ctx.save_for_backward(x, output)
#     return output

#   @staticmethod
#   def backward(ctx, grad_output):
#     x, output = ctx.saved_variables
#     grad_input = None

#     if ctx.needs_input_grad[0]:
#       grad_input = grad_output * output * x.inverse().t()

#     return grad_input

# class LogDet(torch.autograd.Function):
#   """Matrix log determinant. Input should be a square matrix."""

#   @staticmethod
#   def forward(ctx, x):
#     output = torch.potrf(x).diag().prod(dim=0) ** 2
#     ctx.save_for_backward(x, output)
#     return output

#   @staticmethod
#   def backward(ctx, grad_output):
#     x, output = ctx.saved_variables
#     grad_input = None

#     if ctx.needs_input_grad[0]:
#       grad_input = grad_output * output * x.inverse().t()

#     return grad_input
