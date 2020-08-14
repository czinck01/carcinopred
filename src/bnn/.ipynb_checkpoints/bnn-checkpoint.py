import functools as ft

import tensorflow as tf
import tensorflow_probability as tfp


def dense(inputs, weights, biases, activation):
    return activation(tf.matmul(inputs, weights) + biases)

## Changed: num_classes
def build_network(weights_list, biases_list, activation=tf.nn.relu):
    def model(samples, training=True):
        net = samples
        for (weights, biases) in zip(weights_list[:-1], biases_list[:-1]):
            net = dense(net, weights, biases, activation)
        # final linear layer
        net = tf.matmul(net, weights_list[-1]) + biases_list[-1]
        # the model's predictive mean and log variance (each of size samples.shape(0))
        #y_pred, y_log_var = tf.unstack(net, axis=1)
        if training:
            return tfp.distributions.Categorical(logits=net)
            
        else:
            return tf.nn.softmax(net) 

    return model


def bnn_log_prob_fn(X, y, weights, biases, get_mean=False):
    """Compute log likelihood of predicted labels y given the
    features X, weights W and biases b.
    
    Args:
        X (np.array): 2d feature values.
        y (np.array): 1d labels (ground truth).
        weights (list): 2d arrays of weights for each layer.
        biases (list): 1d arrays of biases for each layer.
        get_mean (bool, optional): Whether to return the mean log
        probability over all labels for diagnostics, e.g. to
        compare train and test set performance. Defaults to False.
    
    Returns:
        tf.tensor: Sum or mean of log probabilities of all labels.
    """
    network = build_network(weights, biases)
    labels_dist = network(X)
    if get_mean:
        return tf.reduce_mean(labels_dist.log_prob(y))
    return tf.reduce_sum(labels_dist.log_prob(y))


def prior_log_prob_fn(weight_prior, bias_prior, weights, biases):
    log_prob = sum([tf.reduce_sum(weight_prior.log_prob(w)) for w in weights])
    log_prob += sum([tf.reduce_sum(bias_prior.log_prob(b)) for b in biases])
    return log_prob


def target_log_prob_fn_factory(weight_prior, bias_prior, X_train, y_train):
    def target_log_prob_fn(*args):
        weights, biases = args[::2], args[1::2]
        log_prob = prior_log_prob_fn(weight_prior, bias_prior, weights, biases)
        log_prob += bnn_log_prob_fn(X_train, y_train, weights, biases)
        return log_prob

    return target_log_prob_fn


def tracer_factory(X, y):
    return lambda *args: ft.partial(bnn_log_prob_fn, X, y, get_mean=True)(
        args[::2], args[1::2]
    ).numpy()

#Added num_classes
def get_random_initial_state(weight_prior, bias_prior, nodes_per_layer, overdisp=1.0):
    """Generate random initial configuration for weights and biases of a fully-connected NN
    sampled according to the specified prior distributions. This configuration can serve
    as a starting point for instance to generate a Markov chain of network configurations
    via Hamiltonian Monte Carlo which are distributed according to the posterior after having
    observed some data.
    """
    init_state = []
    for idx in range(len(nodes_per_layer) - 1):
        weights_shape = (nodes_per_layer[idx], nodes_per_layer[idx + 1])
        biases_shape = nodes_per_layer[idx + 1]
        # use overdispersion > 1 for better R-hat statistics
        weights = weight_prior.sample(tf.squeeze(weights_shape)) * overdisp
        biases = bias_prior.sample(tf.squeeze(biases_shape)) * overdisp
        init_state.extend((weights, biases))
    '''#Last layer output logit 
    weights_shape = (nodes_per_layer[-1], num_classes)
    biases_shape = num_classes
    weights = weight_prior.sample(tf.squeeze(weights_shape)) * overdisp
    biases = bias_prior.sample(tf.squeeze(biases_shape)) * overdisp
    init_state.extend((weights, biases))'''
    return init_state