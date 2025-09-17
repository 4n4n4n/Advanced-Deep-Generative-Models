def likelihood(X_train, model, device):
    """
    X_train: shape (B, D)
    model: flows
    """
    ##########################################################
    # YOUR CODE HERE
    # calling model.log_prob
    X_train = X_train.to(device)
    log_prob = model.log_prob(X_train)
    # average 
    loss = 0
    for p in log_prob:
        loss -= p
    loss /= X_train.shape[0]
    ##########################################################

    return loss
