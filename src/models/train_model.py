def train(model, X_train, y_train):
    """
    Train the given model on the training data.
    Handles early stopping for models that support it (Paper §7.3).
    """
    model.fit(X_train, y_train)
    return model