from sklearn.neural_network import MLPClassifier


def get_model():
    """
    Return a Deep Neural Network model (Paper §6.2.4).
    Uses multiple hidden layers with ReLU activation.
    'DNN architectures, operating utilizing multiple hidden perceptron layers
     containing advanced activation functions (such as ReLU and Leaky ReLU),
     form the apex of the classification stack.'
    Implemented via sklearn MLPClassifier for portability.
    """
    return MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),   # 3 hidden layers
        activation='relu',                   # ReLU activation (Paper §6.2.4)
        solver='adam',                        # Adam optimizer
        max_iter=200,
        early_stopping=True,                  # Prevent overfitting (Paper §7.3)
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        verbose=False
    )
