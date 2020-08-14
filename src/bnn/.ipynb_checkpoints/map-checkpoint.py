import numpy as np
import tensorflow as tf

import src.bnn as bnn_fn


def get_map_trace(
    target_log_prob_fn, state, num_iters=1000, save_every=10, callbacks=[]
):
    state_vars = [tf.Variable(s) for s in state]
    optimizer = tf.optimizers.Adam()

    def map_loss():
        return -target_log_prob_fn(*state_vars)

    @tf.function
    def minimize():
        optimizer.minimize(map_loss, state_vars)

    state_trace, cb_trace = [[] for _ in state], [[] for _ in callbacks]
    for i in range(num_iters):
        if i % save_every == 0:
            for trace, state in zip(state_trace, state_vars):
                trace.append(state.numpy())
            for trace, cb in zip(cb_trace, callbacks):
                trace.append(cb(*state_vars))
        minimize()
    return state_trace, cb_trace


def get_best_map_state(map_trace, map_log_probs):
    # map_log_probs contains the log probability
    # trace for both train [0] and test set [1].
    test_set_max_log_prob_idx = np.argmax(map_log_probs[1])
    # We return the MAP NN configuration that achieved the
    # highest likelihood on the test set.
    return [tf.constant(tr[test_set_max_log_prob_idx]) for tr in map_trace]


def get_nodes_per_layer(n_features, net_taper=(1, 0.5, 0.2, 0.1)):
    nodes_per_layer = [int(n_features * x) for x in net_taper]
    # Ensure the last layer has two nodes so that output can be split into
    # predictive mean and learned loss attenuation (see eq. (7) of
    # https://arxiv.org/abs/1703.04977) which the network learns individually.
    nodes_per_layer.append(2)
    return nodes_per_layer


def map_predict(weight_prior, bias_prior, X_train, y_train, X_test, y_test):
    """Generate maximum a posteriori neural network predictions.
    
    Args:
        weight_prior (tfp.distribution): Prior probability for the weights
        bias_prior (tfp.distribution): Prior probability for the biases
        [X/y_train/test] (np.arrays): Train and test sets
    """

    log_prob_tracers = (
        bnn_fn.tracer_factory(X_train, y_train),
        bnn_fn.tracer_factory(X_test, y_test),
    )

    _, n_features = X_train.shape  # number of samples times number of features
    random_initial_state = bnn_fn.get_random_initial_state(
        weight_prior, bias_prior, get_nodes_per_layer(n_features)
    )

    trace, log_probs = get_map_trace(
        bnn_fn.target_log_prob_fn_factory(weight_prior, bias_prior, X_train, y_train),
        random_initial_state,
        num_iters=3000,
        callbacks=log_prob_tracers,
    )
    # Can be used as initial configuration for other methods such as Hamiltonian Monte Carlo.
    best_params = get_best_map_state(trace, log_probs)

    weights, biases = best_params[::2], best_params[1::2]
    model = bnn_fn.build_network(weights, biases)
    y_pred, y_var = model(X_test, training=False)
    return y_pred.numpy(), y_var.numpy(), log_probs, best_params