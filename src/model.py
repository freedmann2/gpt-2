import numpy as np
import tensorflow.compat.v1 as tf
from typing import Any, List, Tuple, Union


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

    def override_from_dict(self, d) -> None:
        self.update(d)


def shapelist(x):
    if hasattr(x, 'shape'):
        x = x.shape
    if hasattr(x, 'as_list'):
        x = x.as_list()
    return x

def default_hparams():
    return EasyDict(
        n_vocab=50257,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
        res_dropout=0.0,
        attn_dropout=0.0,
        dtype=tf.float32
    )

import os

def get_variable(name):
    name = os.path.join(tf.get_variable_scope().name, name)
    vs = tf.trainable_variables()
    for x in vs:
        if x.name.startswith(name + ':'):
            return x

def init_variable(name, shape, **kws):
    v = get_variable(name)
    if v is None:
        v = tf.get_variable(name, shape, **kws)
    v = graph_spectral_norm(v)
    return v


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def norm(x, scope, *, axis=-1, epsilon=1e-5, hparams=None):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        n_state = shape_list(x)[-1]
        g = init_variable('g', [n_state], initializer=tf.constant_initializer(1, dtype=dtype))
        b = init_variable('b', [n_state], initializer=tf.constant_initializer(0, dtype=dtype))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b
        return x

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

def conv1d(x, scope, nf, *, w_init_stdev=0.02, hparams=None):
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        *start, nx = shape_list(x)
        w = init_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev, dtype=dtype))
        b = init_variable('b', [nf], initializer=tf.constant_initializer(0, dtype=dtype))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(shape_list(v)[-1], w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        w = dropout(w, hparams.attn_dropout)
        a = tf.matmul(w, v)
        return a

    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        c = conv1d(x, 'c_attn', n_state*3, hparams=hparams)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, hparams=hparams)
        a = dropout(a, hparams.res_dropout)
        return a, present


def mlp(x, scope, n_state, *, hparams):
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        nx = shape_list(x)[-1]
        h = gelu(conv1d(x, 'c_fc', n_state, hparams=hparams))
        h2 = conv1d(h, 'c_proj', nx, hparams=hparams)
        h2 = dropout(h2, hparams.res_dropout)
        return h2

def dropout(x, pdrop=0.1, train=True):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, rate=pdrop)
    return x

def block(x, scope, *, past, hparams):
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        nx = shape_list(x)[-1]
        a, present = attn(norm(x, 'ln_1', hparams=hparams), 'attn', nx, past=past, hparams=hparams)
        x = x + a
        m = mlp(norm(x, 'ln_2', hparams=hparams), 'mlp', nx*4, hparams=hparams)
        x = x + m
        return x, present

def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


def ff(x, *, hparams):
  nx = shape_list(x)[-1]
  x = norm(x, 'ln_2', hparams=hparams)
  h = mlp(x, 'mlp', nx*4, hparams=hparams)
  return h


def fft(x):
  return tf.signal.dct(x)


def concat(xs, axis):
  return tf.concat(xs, axis=axis)

def attend(y, x, nx, *, past, hparams):
  #*start, nx = shape_list(x)
  # x = norm(x, 'ln_1', hparams=hparams)
  a, present = attn(x, 'attn', nx, past=past, hparams=hparams)
  return a, present

def extended_self_attention_layer(x, *, past, hparams):
  *start, nx = shape_list(x)
  x = norm(x, 'ln_1', hparams=hparams)
  y = fft(x)
  z = concat([x, y], axis=-1)
  #import pdb; pdb.set_trace()
  h, present = attend(y, z, nx=nx, past=past, hparams=hparams) # Attend to the concatenation of both of the previous layers.
  return h, present


def graph_name(name):
  name = name.split(':')[0]
  #name = name.split('/kernel')[0]
  return name

scalars = []


def scalar(name, val, init=None):
  scalars.append(tf.summary.scalar(name, val))


def graph_spectral_norm(w, init=None):
  # if tpu_summaries.TpuSummaries.inst is None:
  #   return w
  name = graph_name(w.name)
  # if name is not None and not tpu_summaries.TpuSummaries.inst.has(name):
  if name is not None:
    tf.logging.info("[ops] Graphing name=%s (was %s), %s", name, w.name, repr(w))
    #w1, norm = spectral_norm(w)
    norm = spectral_norm_stateless(w)
    scalar(name, norm, init=init)
  else:
    tf.logging.info("[ops] Not graphing %s", w.name)
  return w


def spectral_norm_stateless(inputs, epsilon=1e-12, singular_value="right",
                  power_iteration_rounds=5):
  """Performs Spectral Normalization on a weight tensor.

  Details of why this is helpful for GAN's can be found in "Spectral
  Normalization for Generative Adversarial Networks", Miyato T. et al., 2018.
  [https://arxiv.org/abs/1802.05957].

  Args:
    inputs: The weight tensor to normalize.
    epsilon: Epsilon for L2 normalization.
    singular_value: Which first singular value to store (left or right). Use
      "auto" to automatically choose the one that has fewer dimensions.

  Returns:
    The normalized weight tensor.
  """
  if len(inputs.shape) <= 0:
    logging.info("[ops] spectral norm of a float is itself; returning as-is. name=%s %s", inputs.name, repr(inputs))
    return inputs

  # The paper says to flatten convnet kernel weights from (C_out, C_in, KH, KW)
  # to (C_out, C_in * KH * KW). Our Conv2D kernel shape is (KH, KW, C_in, C_out)
  # so it should be reshaped to (KH * KW * C_in, C_out), and similarly for other
  # layers that put output channels as last dimension. This implies that w
  # here is equivalent to w.T in the paper.
  w = tf.reshape(inputs, (-1, inputs.shape[-1]))

  # Choose whether to persist the first left or first right singular vector.
  # As the underlying matrix is PSD, this should be equivalent, but in practice
  # the shape of the persisted vector is different. Here one can choose whether
  # to maintain the left or right one, or pick the one which has the smaller
  # dimension. We use the same variable for the singular vector if we switch
  # from normal weights to EMA weights.
  if singular_value == "auto":
    singular_value = "left" if w.shape[0] <= w.shape[1] else "right"
  u_shape = (w.shape[0], 1) if singular_value == "left" else (1, w.shape[-1])
  u = tf.random.normal(shape=u_shape, name='u0')

  # Use power iteration method to approximate the spectral norm.
  # The authors suggest that one round of power iteration was sufficient in the
  # actual experiment to achieve satisfactory performance.
  for _ in range(power_iteration_rounds):
    if singular_value == "left":
      # `v` approximates the first right singular vector of matrix `w`.
      v = tf.math.l2_normalize(
          tf.matmul(tf.transpose(w), u), axis=None, epsilon=epsilon)
      u = tf.math.l2_normalize(tf.matmul(w, v), axis=None, epsilon=epsilon)
    else:
      v = tf.math.l2_normalize(tf.matmul(u, w, transpose_b=True),
                               epsilon=epsilon)
      u = tf.math.l2_normalize(tf.matmul(v, w), epsilon=epsilon)

  # The authors of SN-GAN chose to stop gradient propagating through u and v
  # and we maintain that option.
  u = tf.stop_gradient(u)
  v = tf.stop_gradient(v)

  if singular_value == "left":
    norm_value = tf.matmul(tf.matmul(tf.transpose(u), w), v)
  else:
    norm_value = tf.matmul(tf.matmul(v, w), u, transpose_b=True)
  norm_value.shape.assert_is_fully_defined()
  norm_value.shape.assert_is_compatible_with([1, 1])
  return norm_value[0][0]



def model(hparams, X, past=None, scope='model', reuse=tf.AUTO_REUSE):
    # import pdb; pdb.set_trace()
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, reuse=reuse, dtype=dtype):
        results = {}
        batch, sequence = shape_list(X)

        wpe = init_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01, dtype=dtype))
        wte = init_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02, dtype=dtype))
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        if True:
            y = None
            for layer, past in enumerate(pasts):
              scope = 'h%d' % layer
              with tf.variable_scope(scope, dtype=dtype):
                with tf.variable_scope('b0', dtype=dtype):
                  new_y0, present = extended_self_attention_layer(h, past=past, hparams=hparams)
                  new_y = ff(new_y0, hparams=hparams)
                  if y is not None:
                    y = new_y + y
                  else:
                    y = new_y
                with tf.variable_scope('b1', dtype=dtype):
                  y0, present = extended_self_attention_layer(y, past=past, hparams=hparams)
                  h = ff(y0, hparams=hparams) + h
                presents.append(present)
        else:
            for layer, past in enumerate(pasts):
                h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
                presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f', hparams=hparams)

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results
