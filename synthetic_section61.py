r"""Synthetic binary experiment from Sec. 6.1 / Fig. 2 (ICLR 2021 logit adjustment).

Setup (paper):
  - y \in {±1}; x | y ~ N(\mu_y, \sigma^2 I) with \mu_y = y \cdot (+1, +1).
  - Class imbalance P(y = +1) = 5% (configurable).
  - Affine classifier (2 logits); pairwise margin loss (11) with \alpha_y = 1.
  - ERM, adaptive (Cao et al. 2019, Eq. 5), equalised (Tan et al. 2020, Eq. 6),
    logit-adjusted (Eq. 10 / (11) with \Delta_{yy'} = \log(\pi_{y'}/\pi_y)).

Example (from this directory):
  python synthetic_section61.py --num_trials=100 --plot
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

FLAGS: Optional[argparse.Namespace] = None


def _build_arg_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
      description='Synthetic binary experiment from Sec. 6.1 / Fig. 2.',
  )
  p.add_argument('--num_train', type=int, default=10000, help='Training samples per trial.')
  p.add_argument('--num_test', type=int, default=10000, help='Test samples per trial.')
  p.add_argument('--num_trials', type=int, default=100, help='Independent trials for bar chart.')
  p.add_argument('--pi_positive', type=float, default=0.05, help='P(y = +1), rare positive class.')
  p.add_argument('--sigma', type=float, default=1.0, help='Isotropic Gaussian std for p(x|y).')
  p.add_argument('--epochs', type=int, default=80, help='Training epochs per trial.')
  p.add_argument('--batch_size', type=int, default=128, help='Minibatch size.')
  p.add_argument('--learning_rate', type=float, default=0.1, help='SGD learning rate.')
  p.add_argument('--momentum', type=float, default=0.9, help='SGD momentum.')
  p.add_argument(
      '--nesterov',
      action=argparse.BooleanOptionalAction,
      default=True,
      help='Use Nesterov momentum.',
  )
  p.add_argument('--seed', type=int, default=0, help='Base RNG seed.')
  p.add_argument('--output_dir', type=str, default='synthetic_fig2', help='Where to save figures.')
  p.add_argument('--plot', action='store_true', help='If set, save matplotlib figures.')
  p.add_argument('--tau_max', type=float, default=2.0, help='Max τ for post-hoc curves (Fig. 2 right).')
  p.add_argument('--tau_steps', type=int, default=41, help='Number of τ values in [0, tau_max].')
  p.add_argument(
      '--model',
      type=str,
      default='linear',
      choices=['linear', 'mlp'],
      help='Classifier type. "linear" matches the paper; "mlp" adds nonlinearity.',
  )
  p.add_argument('--hidden_dim', type=int, default=64, help='MLP hidden width (when --model=mlp).')
  p.add_argument('--hidden_layers', type=int, default=2, help='Number of hidden layers (when --model=mlp).')
  return p


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
  return _build_arg_parser().parse_args(argv)


def _fig_suffix() -> str:
  assert FLAGS is not None
  return '_mlp' if getattr(FLAGS, 'model', 'linear') == 'mlp' else ''


# Class index 0 = paper's y = +1 (rare), index 1 = y = -1 (frequent).
MU = np.array([[1.0, 1.0], [-1.0, -1.0]], dtype=np.float32)


def balanced_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Average of per-class misclassification rates (two classes)."""
  y_true = y_true.astype(np.int32)
  y_pred = y_pred.astype(np.int32)
  m0 = np.mean(y_pred[y_true == 0] != 0) if np.any(y_true == 0) else 0.0
  m1 = np.mean(y_pred[y_true == 1] != 1) if np.any(y_true == 1) else 0.0
  return float(0.5 * (m0 + m1))


def bayes_predict(x: np.ndarray) -> np.ndarray:
  """Bayes-optimal decision for balanced error: P(x|+1) > P(x|-1) iff (μ_+ - μ_-)^T x > 0."""
  # Label 0 = y=+1 (rare). Appendix F: predict +1 iff (μ_{+1}-μ_{-1})^T x > 0 ⇔ x1+x2 > 0.
  return np.where(x[:, 0] + x[:, 1] > 0, 0, 1).astype(np.int32)

def sample_gaussian_mixture(
    n: int,
    pi_positive: float,
    sigma: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
  """Sample (x, y) with y=0 for +1 class (prob pi_positive), y=1 for -1."""
  u = rng.random(n)
  y = (u >= pi_positive).astype(np.int32)
  means = MU[y]
  x = means + rng.standard_normal((n, 2)).astype(np.float32) * sigma
  return x, y


def margin_deltas(loss_kind: int, pi_positive: float) -> Tuple[float, float]:
  """Returns (Δ_{0,1}, Δ_{1,0}) for loss (11) with α_y = 1 (Sec. 5.2, Sec. 6.1)."""
  pi0 = max(pi_positive, 1e-12)
  pi1 = max(1.0 - pi_positive, 1e-12)
  if loss_kind == 0:  # ERM
    return 0.0, 0.0
  if loss_kind == 1:  # Adaptive (Cao et al., Eq. 5): e^{δ_y} with δ_y ∝ π_y^{-1/4}
    return float(pi0**-0.25), float(pi1**-0.25)
  if loss_kind == 2:  # Equalised (Tan et al., Eq. 6): e^{δ_{y'}}
    return float(np.log(pi1)), float(np.log(pi0))
  if loss_kind == 3:  # Logit adjusted (Eq. 10): Δ_{yy'} = log(π_{y'}/π_y)
    return float(np.log(pi1 / pi0)), float(np.log(pi0 / pi1))
  raise ValueError(f'Unknown loss_kind {loss_kind}')


@tf.function
def pairwise_margin_loss(
    labels: tf.Tensor,
    logits: tf.Tensor,
    d01: tf.Tensor,
    d10: tf.Tensor,
) -> tf.Tensor:
  """Loss (11) with α_y = 1: mean_y softplus(Δ_{y,y'} + f_{y'} - f_y)."""
  oh = tf.one_hot(labels, 2, dtype=tf.float32)
  f_y = tf.reduce_sum(oh * logits, axis=1)
  oh_o = tf.one_hot(1 - labels, 2, dtype=tf.float32)
  f_o = tf.reduce_sum(oh_o * logits, axis=1)
  yf = tf.cast(labels, tf.float32)
  delta = (1.0 - yf) * d01 + yf * d10
  return tf.reduce_mean(tf.nn.softplus(delta + f_o - f_y))


def train_affine_one_trial(
    x_train: np.ndarray,
    y_train: np.ndarray,
    loss_kind: int,
    pi_positive: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    nesterov: bool,
) -> tf.keras.Model:
  """Train 2-class model (linear or MLP) with softmax margin loss."""
  d01_f, d10_f = margin_deltas(loss_kind, pi_positive)
  d01 = tf.constant(d01_f, dtype=tf.float32)
  d10 = tf.constant(d10_f, dtype=tf.float32)

  layers = [tf.keras.layers.Input(shape=(2,))]
  if getattr(FLAGS, 'model', 'linear') == 'mlp':
    h = int(getattr(FLAGS, 'hidden_dim', 64))
    L = int(getattr(FLAGS, 'hidden_layers', 2))
    for _ in range(max(L, 1)):
      layers.append(tf.keras.layers.Dense(h, activation='relu', use_bias=True))
  layers.append(tf.keras.layers.Dense(2, use_bias=True))
  model = tf.keras.Sequential(layers)
  opt = tf.keras.optimizers.SGD(
      learning_rate=learning_rate,
      momentum=momentum,
      nesterov=nesterov,
  )

  @tf.function
  def train_step(xb, yb):
    with tf.GradientTape() as tape:
      logits = model(xb, training=True)
      loss = pairwise_margin_loss(yb, logits, d01, d10)
    g = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(g, model.trainable_variables))
    return loss

  for _ in range(epochs):
    idx = np.random.permutation(len(x_train))
    x_s = x_train[idx]
    y_s = y_train[idx]
    ds = tf.data.Dataset.from_tensor_slices((x_s, y_s)).batch(batch_size)
    for xb, yb in ds:
      train_step(xb, yb)
  return model


def predict_erm(logits: np.ndarray) -> np.ndarray:
  return np.argmax(logits, axis=1).astype(np.int32)


def predict_logit_adjusted(
    logits: np.ndarray, pi: np.ndarray, tau: np.ndarray
) -> np.ndarray:
  """Eq. (9): argmax_y f_y - τ log π_y (τ can be scalar or per-row for broadcasting)."""
  adj = logits - tau * np.log(pi + 1e-12)
  return np.argmax(adj, axis=1).astype(np.int32)


def predict_weight_norm(
    logits: np.ndarray, pi: np.ndarray, tau: np.ndarray
) -> np.ndarray:
  """Eq. (3) with ν_y = π_y: argmax_y f_y / π_y^τ."""
  scale = np.power(pi + 1e-12, tau)
  adj = logits / scale
  return np.argmax(adj, axis=1).astype(np.int32)


def run_bar_chart_trials() -> Tuple[np.ndarray, float]:
  """Returns errors shape (num_trials, 5) for erm, adaptive, equalised, logit_adj, bayes; and bayes mean."""
  rng = np.random.default_rng(FLAGS.seed)
  pi_p = FLAGS.pi_positive
  kinds = [0, 1, 2, 3]
  names = ['erm', 'adaptive', 'equalised', 'logit_adj']
  errors = np.zeros((FLAGS.num_trials, len(kinds) + 1), dtype=np.float64)

  for t in range(FLAGS.num_trials):
    tr = np.random.SeedSequence((FLAGS.seed, t)).generate_state(1)[0]
    r = np.random.default_rng(tr)
    x_tr, y_tr = sample_gaussian_mixture(
        FLAGS.num_train, pi_p, FLAGS.sigma, r)
    x_te, y_te = sample_gaussian_mixture(
        FLAGS.num_test, pi_p, FLAGS.sigma, r)

    bayes_te = bayes_predict(x_te)
    errors[t, 4] = balanced_error(y_te, bayes_te)

    for k, kind in enumerate(kinds):
      model = train_affine_one_trial(
          x_tr,
          y_tr,
          kind,
          pi_p,
          FLAGS.epochs,
          FLAGS.batch_size,
          FLAGS.learning_rate,
          FLAGS.momentum,
          FLAGS.nesterov,
      )
      logits = model.predict(x_te, verbose=0)
      pred = predict_erm(logits)
      errors[t, k] = balanced_error(y_te, pred)

  return errors, float(np.mean(errors[:, 4]))


def run_posthoc_curves(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """After ERM training, sweep τ for logit adjustment vs weight normalisation."""
  pi_p = FLAGS.pi_positive
  pi = np.array([pi_p, 1.0 - pi_p], dtype=np.float32)

  model = train_affine_one_trial(
      x_train,
      y_train,
      0,
      pi_p,
      FLAGS.epochs,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.momentum,
      FLAGS.nesterov,
  )
  logits = model.predict(x_test, verbose=0)

  taus = np.linspace(0.0, FLAGS.tau_max, FLAGS.tau_steps, dtype=np.float64)
  err_la = np.zeros_like(taus)
  err_wn = np.zeros_like(taus)
  for i, tau in enumerate(taus):
    err_la[i] = balanced_error(
        y_test, predict_logit_adjusted(logits, pi, tau))
    err_wn[i] = balanced_error(
        y_test, predict_weight_norm(logits, pi, tau))
  return taus, err_la, err_wn


def plot_fig2_left_balanced_error(errors: np.ndarray, bayes_mean: float, plt) -> str:
  names = ['ERM', 'Adaptive', 'Equalised', 'Logit adj.', 'Bayes']
  means = np.mean(errors, axis=0) * 100.0
  stds = np.std(errors, axis=0) * 100.0

  fig, ax = plt.subplots(figsize=(6, 4))
  xpos = np.arange(4)
  ax.bar(xpos, means[:4], yerr=stds[:4], capsize=3, color=['#5c7bd9', '#9b7ed9', '#d97e7e', '#7ed9a8'])
  ax.axhline(bayes_mean * 100.0, color='k', linestyle='--', label='Bayes (mean over trials)')
  ax.set_xticks(xpos)
  ax.set_xticklabels(names[:4], rotation=15, ha='right')
  ax.set_ylabel('Balanced error (%)')
  ax.set_title(
      f'Sec. 6.1 synthetic (π_+ = {FLAGS.pi_positive}, σ = {FLAGS.sigma})')
  ax.legend(loc='upper right')
  fig.tight_layout()

  out_path = os.path.join(
      FLAGS.output_dir, f'fig2_left_balanced_error{_fig_suffix()}.png')
  fig.savefig(out_path, dpi=150)
  plt.close(fig)
  return out_path


def _boundary_normal(w: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
  # f_0 - f_1 = 0 => (w0-w1)^T x + (b0-b1) = 0
  w0, w1 = w[:, 0], w[:, 1]
  n = w0 - w1
  off = b[0] - b[1]
  return float(n[0]), float(n[1]), float(off)


def _line_from_normal(
    a: float, b: float, c: float, xgrid: np.ndarray
) -> np.ndarray | None:
  # a*x + b*y + c = 0 -> y = (-a*x - c) / b
  if abs(b) < 1e-8:
    return None
  return (-a * xgrid - c) / b


def plot_fig2_middle_separators(plt) -> str:
  # Middle-style: one trial scatter + decision boundaries (ERM vs logit-adj vs Bayes)
  rng = np.random.default_rng(FLAGS.seed + 999)
  x_tr, y_tr = sample_gaussian_mixture(
      FLAGS.num_train, FLAGS.pi_positive, FLAGS.sigma, rng)
  x_te, y_te = sample_gaussian_mixture(2000, FLAGS.pi_positive, FLAGS.sigma, rng)

  model_erm = train_affine_one_trial(
      x_tr, y_tr, 0, FLAGS.pi_positive, FLAGS.epochs, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.momentum, FLAGS.nesterov)
  model_adapt = train_affine_one_trial(
      x_tr, y_tr, 1, FLAGS.pi_positive, FLAGS.epochs, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.momentum, FLAGS.nesterov)
  model_eq = train_affine_one_trial(
      x_tr, y_tr, 2, FLAGS.pi_positive, FLAGS.epochs, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.momentum, FLAGS.nesterov)
  model_la = train_affine_one_trial(
      x_tr, y_tr, 3, FLAGS.pi_positive, FLAGS.epochs, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.momentum, FLAGS.nesterov)

  w_erm, b_erm = model_erm.layers[-1].get_weights()
  w_adapt, b_adapt = model_adapt.layers[-1].get_weights()
  w_eq, b_eq = model_eq.layers[-1].get_weights()
  w_la, b_la = model_la.layers[-1].get_weights()

  fig, ax = plt.subplots(figsize=(5, 5))
  c0 = y_te == 0
  c1 = y_te == 1
  ax.scatter(x_te[c0, 0], x_te[c0, 1], s=6, alpha=0.35, c='C0', label='y=+1 (rare)')
  ax.scatter(x_te[c1, 0], x_te[c1, 1], s=6, alpha=0.35, c='C1', label='y=-1')
  xs = np.linspace(-4.0, 4.0, 200)

  # Bayes: x + y = 0
  ax.plot(xs, -xs, 'k--', lw=2, label='Bayes (x1+x2=0)')

  a, b, c = _boundary_normal(w_erm, b_erm)
  yl = _line_from_normal(a, b, c, xs)
  if yl is not None:
    ax.plot(xs, yl, '-', lw=1.5, label='ERM separator')

  a, b, c = _boundary_normal(w_adapt, b_adapt)
  yl = _line_from_normal(a, b, c, xs)
  if yl is not None:
    ax.plot(xs, yl, '-', lw=1.5, label='Adaptive separator')

  a, b, c = _boundary_normal(w_eq, b_eq)
  yl = _line_from_normal(a, b, c, xs)
  if yl is not None:
    ax.plot(xs, yl, '-', lw=1.5, label='Equalised separator')

  a, b, c = _boundary_normal(w_la, b_la)
  yl = _line_from_normal(a, b, c, xs)
  if yl is not None:
    ax.plot(xs, yl, '-', lw=1.5, label='Logit-adj. loss separator')

  ax.set_xlim(-4, 4)
  ax.set_ylim(-4, 4)
  ax.set_aspect('equal')
  ax.set_xlabel(r'$x_1$')
  ax.set_ylabel(r'$x_2$')
  ax.legend(loc='upper left', fontsize=8)
  ax.set_title('Learned linear separators (one trial)')
  fig.tight_layout()

  out_path = os.path.join(
      FLAGS.output_dir, f'fig2_middle_separators{_fig_suffix()}.png')
  fig.savefig(out_path, dpi=150)
  plt.close(fig)
  return out_path


def plot_fig2_right_posthoc(
    taus: np.ndarray,
    err_la: np.ndarray,
    err_wn: np.ndarray,
    bayes_test: float,
    plt,
) -> str:
  fig, ax = plt.subplots(figsize=(5.5, 4))
  ax.plot(taus, err_la * 100.0, label='Post-hoc logit adjustment', lw=2)
  ax.plot(taus, err_wn * 100.0, label='Weight normalisation (ν=π)', lw=2)
  ax.axhline(bayes_test * 100.0, color='k', linestyle='--', label='Bayes (test)')
  ax.set_xlabel(r'Scaling $\tau$')
  ax.set_ylabel('Balanced error (%)')
  ax.set_title('Post-hoc adjustment vs τ (ERM-trained model)')
  ax.legend()
  fig.tight_layout()

  out_path = os.path.join(
      FLAGS.output_dir, f'fig2_right_posthoc{_fig_suffix()}.png')
  fig.savefig(out_path, dpi=150)
  plt.close(fig)
  return out_path


def maybe_plot(
    errors: np.ndarray,
    bayes_mean: float,
    taus: np.ndarray,
    err_la: np.ndarray,
    err_wn: np.ndarray,
    bayes_test: float,
) -> None:
  try:
    import matplotlib.pyplot as plt
  except ImportError:
    print('matplotlib not installed; skip --plot. pip install matplotlib')
    return

  os.makedirs(FLAGS.output_dir, exist_ok=True)

  p1 = plot_fig2_left_balanced_error(errors, bayes_mean, plt)
  print(f'Wrote {p1}')
  p2 = plot_fig2_middle_separators(plt)
  print(f'Wrote {p2}')
  p3 = plot_fig2_right_posthoc(taus, err_la, err_wn, bayes_test, plt)
  print(f'Wrote {p3}')


def main(argv: Optional[list[str]] = None) -> None:
  global FLAGS
  FLAGS = _parse_args(argv)

  tf.keras.utils.set_random_seed(FLAGS.seed)

  print('Running Sec. 6.1 bar-chart trials (training losses: ERM, adaptive, equalised, logit adj.)...')
  errors, bayes_mean = run_bar_chart_trials()
  print(
      'Mean balanced error (%): '
      f'ERM={np.mean(errors[:, 0])*100:.3f}, '
      f'Adaptive={np.mean(errors[:, 1])*100:.3f}, '
      f'Equalised={np.mean(errors[:, 2])*100:.3f}, '
      f'Logit-adj={np.mean(errors[:, 3])*100:.3f}, '
      f'Bayes={bayes_mean*100:.3f}'
  )
  print(
      'Std balanced error (%): '
      f'ERM={np.std(errors[:, 0])*100:.3f}, '
      f'Adaptive={np.std(errors[:, 1])*100:.3f}, '
      f'Equalised={np.std(errors[:, 2])*100:.3f}, '
      f'Logit-adj={np.std(errors[:, 3])*100:.3f}'
  )

  rng = np.random.default_rng(FLAGS.seed + 1)
  x_tr, y_tr = sample_gaussian_mixture(
      FLAGS.num_train, FLAGS.pi_positive, FLAGS.sigma, rng)
  x_te, y_te = sample_gaussian_mixture(
      FLAGS.num_test, FLAGS.pi_positive, FLAGS.sigma, rng)
  bayes_te = balanced_error(y_te, bayes_predict(x_te))

  print('Post-hoc τ sweep (ERM-trained classifier)...')
  taus, err_la, err_wn = run_posthoc_curves(x_tr, y_tr, x_te, y_te)
  ib = int(np.argmin(err_la))
  print(
      f'Best post-hoc logit adj. τ={taus[ib]:.3f}, '
      f'balanced err={err_la[ib]*100:.3f}% (Bayes on test={bayes_te*100:.3f}%)'
  )

  if FLAGS.plot:
    maybe_plot(errors, bayes_mean, taus, err_la, err_wn, bayes_te)


if __name__ == '__main__':
  main()
