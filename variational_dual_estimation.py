"""Filtering algorithms."""

import torch
from torch.autograd import Variable

from utils import nans_tensor, re_triu, size_splits


def agglomerate(ts):
  return torch.cat([t.view(-1) for t in ts])

def diag_cov_mvn_entropy(s):
  """Calculate the entropy of a multivariate normal distribution given the log
  standard deviations up to an additive constant (which it turns out is
  `N / 2 * log(2 * pi * e)`).

  Parameters
  ==========
  s : torch.Tensor
    The log standard deviations.
  """
  return torch.sum(s)

def build_conditional_filter(
    x_dim,
    emission_log_prob,
    transition_log_prob,
    net
):
  """Construct a online dual estimation filter as defined by Algorithm 1 in
  https://arxiv.org/abs/1707.09049. `s_t` is not quite the same thing because
  somehow they're taking a sqrt of theirs which doesn't make much sense in
  general. In this case it denotes the log standard deviation.

  Parameters
  ==========
  x_dim : int
    The dimension of the latent space, ie. the dimension of x_t.
  emission_log_prob : function
  transition_log_prob : function
  net : torch something
  """
  def filter_step(
      y_t,
      u_t,
      u_prev,
      mu_prev,
      s_prev,
      mc_samples=1
  ):
    net_out = net(agglomerate([y_t, u_prev, mu_prev, s_prev]))
    mu_t, s_t = torch.split(net_out, x_dim)

    eps_t = Variable(torch.randn(mc_samples, x_dim), requires_grad=False)
    eps_prev = Variable(torch.randn(mc_samples, x_dim), requires_grad=False)

    # Precompute for speeeeeeeed
    x_squiggle_t = mu_t + torch.exp(s_t) * eps_t
    x_squiggle_prev = mu_prev + torch.exp(s_prev) * eps_prev

    # Instead of dividing the sums by mc_samples, we can just multiply this term
    # by mc_samples.
    L = diag_cov_mvn_entropy(s_t)
    for i in range(mc_samples):
      # x_squiggle_t = mu_t + torch.exp(s_t) * eps_t[i]
      # x_squiggle_prev = mu_prev + torch.exp(s_prev) * eps_prev[i]

      L += 1.0 / mc_samples * (
        emission_log_prob(y_t, x_squiggle_t[i], u_t) +
        transition_log_prob(x_squiggle_t[i], x_squiggle_prev[i], u_prev)
      )

    return mu_t, s_t, -1 * L

  return filter_step

def build_fixed_lag_smoothing_filter(
    x_dim,
    lag,
    emission_log_prob,
    transition_log_prob,
    net
):
  """TODO

  Parameters
  ==========
  x_dim : int
    The dimension of the latent space, ie. the dimension of x_t.
  lag : int
    The number of steps to do fixed-lag smoothing on.
  emission_log_prob : function
  transition_log_prob : function
  net : torch something
  """
  def filter_step(
      y_ts,
      u_ts,
      u_prev,
      mu_prev,
      A_prev,
      mc_samples=1
  ):
    """
    Parameters
    ==========
    y_ts : torch.autograd.Variable of shape [lag x y_dim]
      The observed values for the past lag points.
    u_ts : torch.autograd.Variable of shape [(lag - 1) x u_dim]
      The control values for the past lag time points excluding u_t.
    u_prev : torch.autograd.Variable of shape [u_dim]
      The control value for the time step immediately preceeding this batch of
      time points. In other words, u_{t - lag - 1}.
    mu_prev : torch.autograd.Variable of shape [x_dim]
      The mean of the (old) variational approximation to the filtering
      distribution of the time step immediately preceeding this batch of time
      points. In other words, mu_{t - lag - 1}.
    A_prev : torch.autograd.Variable of shape [x_dim x x_dim]
      The decomposed precision matrix of the (old) variational approximation to
      the filtering distribution of the time step immediately preceeding this
      batch of time points. In other words, A_{t - lag - 1}.
    """
    triu_size = x_dim * (x_dim + 1) / 2

    # TODO: I don't think u_t is really relevant to smoothing these but oh well.
    # The rest of `u_ts` is relevant, so it's easier to just throw the whole
    # thing in.
    net_out = net(agglomerate([y_ts, u_ts, u_prev, mu_prev, A_prev]))
    _mu_ts, _A_diag, _A_offdiag = size_splits(net_out, [
      lag * x_dim,                          # mean
      lag * triu_size,                      # block diagonal triangular matrices
      (lag - 1) * x_dim * x_dim             # block off-diagonal submatrices
    ])

    mu_ts = _mu_ts.view(lag, x_dim)

    # A_diag has shape [lag x x_dim x x_dim]
    A_diag = torch.stack([
      re_triu(vec, x_dim)
      for vec in torch.split(_A_diag, triu_size)
    ])
    # A_offdiag has shape [(lag - 1) x x_dim x x_dim]
    A_offdiag = _A_offdiag.view(lag - 1, x_dim, x_dim)

    def sample_smoothed():
      """Sample from the variational approximation to the fixed-lag smoothing
      posterior."""
      s = nans_tensor(lag, x_dim)
      z = Variable(torch.randn(lag, x_dim), requires_grad=False)
      s[lag - 1] = torch.inverse(A_diag[lag - 1]) @ z[lag - 1]
      for t in reversed(range(lag - 1)):
        s[t] = torch.inverse(A_diag[t]) @ (z[t] - A_offdiag[t] @ s[t + 1])
      return mu_ts + s

    def sample_prev():
      """Sample from the variatioanl approximation to the filtering distribution
      at x_{t - lag - 1}."""
      z = Variable(torch.randn(x_dim), requires_grad=False)
      return mu_prev + torch.inverse(A_prev) @ z

    # The entropy of the variational approximation is the negative sum of the
    # log determinants of the submatrices on the block diagonal. This is because
    # A A^T = cov^-1 more or less. And these submatrices are upper triangular,
    # so this is easy.
    L = -sum(torch.trace(torch.log(A_diag[t])) for t in range(lag))
    for _ in range(mc_samples):
      x_prev = sample_prev()
      x_ts = sample_smoothed()

      L += 1.0 / mc_samples * (
        sum(emission_log_prob(y_ts[t], x_ts[t], u_ts[t]) for t in range(lag)) +
        transition_log_prob(x_ts[0], x_prev, u_prev) +
        sum(transition_log_prob(x_ts[t], x_ts[t - 1], u_ts[t - 1])
            for t in range(1, lag))
      )

    return mu_ts, A_diag, A_offdiag, -1 * L

  return filter_step

def run_conditional_filter(
    filter_step,
    forward_sample,
    optimizer,
    num_steps,
    num_sequences,
    x_dim,
    y_dim,
    u_dim,
    control=None,
    callback=None,
    num_gradient_steps=1
):
  """Run a filter on a given model and update parameters along the way. Supports
  multiple sequences. Uses the L(y_t | y_{1:t-1}) objective.

  Parameters
  ==========
  filter_step : function
    The filtering function which should accept parameters
      y_t : the current observation
      u_t : the current control
      u_prev : the previous control
      mu_prev : the previous posterior mean
      s_prev : the previous posterior log variance
    and return
      mu_t : the estimated posterior mean
      s_t : the estimated posterior log variance
      loss : the variational loss, as in equation 8
  forward_sample : function
    The model sampling function which should accept parameters
      x_prev : the previous latent state
      u_prev : the previous control
    and return
      x_t : the new latent state
      y_t : the new observed state
  optimizer : torch.optim.Optimizer
    The optimizer to run on the combined loss after every time step.
  num_steps : int
    The number of steps to run the filtering algorithm
  num_sequences : int
    The number of independent sequences to filter in parallel, sharing
    parameters
  x_dim : int
    Dimension of the latent state.
  y_dim : int
    Dimension of the observed state.
  u_dim : int
    Dimension of the control.
  control : torch.FloatTensor of shape (num_sequences x num_steps x u_dim), optional
    The control inputs for each of the sequences. Defaults to zeros.
  callback : function, optional
    A callback function to run after every time step. It's results are returned
    as `callback_log`.

  Returns
  =======
  Xs : torch.FloatTensor of shape (num_sequences x num_steps x x_dim)
    The true latent states.
  Ys : torch.FloatTensor of shape (num_sequences x num_steps x y_dim)
    The observations.
  Mus : torch.FloatTensor of shape (num_sequences x num_steps x x_dim)
    The filtered posterior means.
  Ss : torch.FloatTensor of shape (num_sequences x num_steps x x_dim)
    The filtered posterior log variances.
  callback_log : list
  """
  Xs = nans_tensor(num_sequences, num_steps, x_dim)
  Ys = nans_tensor(num_sequences, num_steps, y_dim)
  Us = torch.zeros(num_sequences, num_steps, u_dim)
  if torch.is_tensor(control):
    Us = control

  Mus = nans_tensor(num_sequences, num_steps, x_dim)
  Ss = nans_tensor(num_sequences, num_steps, x_dim)

  # Initialize things to zero at time step 0
  Xs[:, 0, :] = 0
  Ys[:, 0, :] = 0
  Mus[:, 0, :] = 0
  Ss[:, 0, :] = 0

  callback_log = ([] if callback else None)

  for t in range(1, num_steps):
    for _ in range(num_gradient_steps):
      loss = 0.0
      for i in range(num_sequences):
        x_prev = Variable(Xs[i, t - 1], requires_grad=False)
        u_prev = Variable(Us[i, t - 1], requires_grad=False)
        mu_prev = Variable(Mus[i, t - 1], requires_grad=False)
        s_prev = Variable(Ss[i, t - 1], requires_grad=False)
        u_t = Variable(Us[i, t], requires_grad=False)

        # Sample from the true model
        x_t, y_t = forward_sample(x_prev, u_prev)

        mu_t, s_t, loss_i = filter_step(
          y_t,
          u_t,
          u_prev,
          mu_prev,
          s_prev
        )

        # Add the loss for this sequence into the motherload loss
        loss += loss_i

        Xs[i, t] = x_t.data
        Ys[i, t] = y_t.data
        Us[i, t] = u_t.data
        Mus[i, t] = mu_t.data
        Ss[i, t] = s_t.data

      # Finally take the gradient step on the motherload loss
      if optimizer:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if callback:
      info = {
        't': t,
        'loss': loss
      }
      callback_log.append(callback(info))

  return Xs, Ys, Mus, Ss, callback_log

def run_conditional_joint_filter(
    filter_step,
    forward_sample,
    optimizer,
    num_steps,
    num_sequences,
    x_dim,
    y_dim,
    u_dim,
    control=None,
    callback=None
):
  """Run a filter on a given model and update parameters along the way. Supports
  multiple sequences. Uses a lower bound on log p(y_{1:t}) by summing all of the
  conditional losses.

  Parameters
  ==========
  filter_step : function
    The filtering function which should accept parameters
      y_t : the current observation
      u_t : the current control
      u_prev : the previous control
      mu_prev : the previous posterior mean
      s_prev : the previous posterior log variance
    and return
      mu_t : the estimated posterior mean
      s_t : the estimated posterior log variance
      loss : the variational loss, as in equation 8
  forward_sample : function
    The model sampling function which should accept parameters
      x_prev : the previous latent state
      u_prev : the previous control
    and return
      x_t : the new latent state
      y_t : the new observed state
  optimizer : torch.optim.Optimizer
    The optimizer to run on the combined loss after every time step.
  num_steps : int
    The number of steps to run the filtering algorithm
  num_sequences : int
    The number of independent sequences to filter in parallel, sharing
    parameters
  x_dim : int
    Dimension of the latent state.
  y_dim : int
    Dimension of the observed state.
  u_dim : int
    Dimension of the control.
  control : torch.FloatTensor of shape (num_sequences x num_steps x u_dim), optional
    The control inputs for each of the sequences. Defaults to zeros.
  callback : function, optional
    A callback function to run after every time step. It's results are returned
    as `callback_log`.

  Returns
  =======
  Xs : torch.FloatTensor of shape (num_sequences x num_steps x x_dim)
    The true latent states.
  Ys : torch.FloatTensor of shape (num_sequences x num_steps x y_dim)
    The observations.
  Mus : torch.FloatTensor of shape (num_sequences x num_steps x x_dim)
    The filtered posterior means.
  Ss : torch.FloatTensor of shape (num_sequences x num_steps x x_dim)
    The filtered posterior log variances.
  callback_log : list
  """
  Xs = nans_tensor(num_sequences, num_steps, x_dim)
  Ys = nans_tensor(num_sequences, num_steps, y_dim)
  Us = torch.zeros(num_sequences, num_steps, u_dim)
  if torch.is_tensor(control):
    Us = control

  Mus = nans_tensor(num_sequences, num_steps, x_dim)
  Ss = nans_tensor(num_sequences, num_steps, x_dim)

  # Initialize things to zero at time step 0
  Xs[:, 0, :] = 0
  Ys[:, 0, :] = 0
  Mus[:, 0, :] = 0
  Ss[:, 0, :] = 0

  callback_log = ([] if callback else None)

  # Note: This is L(y_{1:t}) not L(y_t | y_{1:t-1})
  loss = 0.0

  for t in range(1, num_steps):
    for i in range(num_sequences):
      x_prev = Variable(Xs[i, t - 1], requires_grad=False)
      u_prev = Variable(Us[i, t - 1], requires_grad=False)
      mu_prev = Variable(Mus[i, t - 1], requires_grad=False)
      s_prev = Variable(Ss[i, t - 1], requires_grad=False)
      u_t = Variable(Us[i, t], requires_grad=False)

      # Sample from the true model
      x_t, y_t = forward_sample(x_prev, u_prev)

      mu_t, s_t, loss_i = filter_step(
        y_t,
        u_t,
        u_prev,
        mu_prev,
        s_prev
      )

      # Add the loss for this sequence into the motherload loss
      loss += loss_i

      Xs[i, t] = x_t.data
      Ys[i, t] = y_t.data
      Us[i, t] = u_t.data
      Mus[i, t] = mu_t.data
      Ss[i, t] = s_t.data

    # Finally take the gradient step on the motherload loss
    if optimizer:
      optimizer.zero_grad()
      loss.backward(retain_graph=True)
      optimizer.step()

    if callback:
      info = {
        't': t,
        'loss': loss
      }
      callback_log.append(callback(info))

  return Xs, Ys, Mus, Ss, callback_log

def run_fixed_lag_smoothing_filter(
    filter_step,
    forward_sample,
    optimizer,
    lag,
    num_steps,
    num_sequences,
    x_dim,
    y_dim,
    u_dim,
    control=None,
    callback=None
):
  """TODO

  Parameters
  ==========
  filter_step : function
    The filtering function which should accept parameters
      y_ts
      u_ts
      u_prev
      mu_prev
      A_prev
    and return
      mu_ts
      A_diag
      A_offdiag
      loss
    See the `build_fixed_lag_smoothing_filter` docs for more info.
  forward_sample : function
    The model sampling function which should accept parameters
      x_prev : the previous latent state
      u_prev : the previous control
    and return
      x_t : the new latent state
      y_t : the new observed state
  optimizer : torch.optim.Optimizer
    The optimizer to run on the combined loss after every time step.
  lag : int
    The number of fixed-lag smoothing steps.
  num_steps : int
    The number of steps to run the filtering algorithm
  num_sequences : int
    The number of independent sequences to filter in parallel, sharing
    parameters
  x_dim : int
    Dimension of the latent state.
  y_dim : int
    Dimension of the observed state.
  u_dim : int
    Dimension of the control.
  control : torch.FloatTensor of shape (num_sequences x num_steps x u_dim), optional
    The control inputs for each of the sequences. Defaults to zeros.
  callback : function, optional
    A callback function to run after every time step. It's results are returned
    as `callback_log`.

  Returns
  =======
  TODO
  """
  Xs = nans_tensor(num_sequences, num_steps, x_dim)
  Ys = nans_tensor(num_sequences, num_steps, y_dim)
  Us = torch.zeros(num_sequences, num_steps, u_dim)
  if torch.is_tensor(control):
    Us = control

  # The mean of the variational filtering estimate over time.
  Mus = nans_tensor(num_sequences, num_steps, x_dim)

  # The precision matrix of the variational filtering estimate over time,
  # decomposed. In other words, A^T * A = Q = cov^-1 and A is upper triangular.
  As = nans_tensor(num_sequences, num_steps, x_dim, x_dim)

  # Initialize things to zero at time step 0
  Xs[:, 0, :] = 0

  # Here we assume that the initial distribution p(x_0) is just constant zero.
  Mus[:, 0, :] = 0
  As[:, 0, :, :] = 0

  callback_log = ([] if callback else None)

  for t in range(1, num_steps):
    loss = 0.0
    for i in range(num_sequences):
      # Sample from the true model
      x_t, y_t = forward_sample(Xs[i, t - 1], Us[i, t - 1])
      Xs[i, t] = x_t.data
      Ys[i, t] = y_t.data

      # If we have enough data now to start doing fixed-lag smoothing then we go
      # ahead and do so.
      if t - lag >= 0:
        u_prev = Variable(Us[i, t - lag], requires_grad=False)
        mu_prev = Variable(Mus[i, t - lag], requires_grad=False)
        A_prev = Variable(As[i, t - lag], requires_grad=False)

        mu, A_diag, _, loss_i = filter_step(
          Ys[i, (t - lag + 1):(t + 1)],
          Us[i, (t - lag + 1):t],
          u_prev,
          mu_prev,
          A_prev
        )

        # Add the loss for this sequence into the motherload loss
        loss += loss_i

        Mus[i, t] = mu.data
        As[i, t] = A_diag[-1].data

    # Finally take the gradient step on the motherload loss
    if optimizer:
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    if callback:
      info = {
        't': t,
        'loss': loss
      }
      callback_log.append(callback(info))

  return Xs, Ys, Mus, As, callback_log
