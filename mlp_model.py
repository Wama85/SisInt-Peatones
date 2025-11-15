import numpy as np

def sigmoid(z):
    """Función de activación sigmoid"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def relu(z):
    """Función de activación ReLU"""
    return np.maximum(0, z)

def relu_der(z):
    """Derivada de ReLU"""
    return (z > 0).astype(float)

def initialize_params(n_x, n_h1, n_h2, n_y):
    """
    Inicializa parámetros para red de 3 capas
    
    Args:
        n_x: número de features de entrada
        n_h1: neuronas en primera capa oculta
        n_h2: neuronas en segunda capa oculta
        n_y: neuronas de salida
    
    Returns:
        W1, b1, W2, b2, W3, b3
    """
    W1 = np.random.randn(n_h1, n_x) * np.sqrt(2/n_x)
    b1 = np.zeros((n_h1, 1))
    W2 = np.random.randn(n_h2, n_h1) * np.sqrt(2/n_h1)
    b2 = np.zeros((n_h2, 1))
    W3 = np.random.randn(n_y, n_h2) * np.sqrt(2/n_h2)
    b3 = np.zeros((n_y, 1))
    return W1, b1, W2, b2, W3, b3

def forward(X, W1, b1, W2, b2, W3, b3):
    """
    Propagación hacia adelante (3 capas)
    
    Args:
        X: datos de entrada (n_x, m)
        W1, b1, W2, b2, W3, b3: parámetros de la red
    
    Returns:
        Z1, A1, Z2, A2, Z3, A3
    """
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = relu(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = sigmoid(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def compute_cost(A3, Y):
    """
    Calcula el costo de entropía cruzada
    
    Args:
        A3: predicciones (1, m)
        Y: etiquetas verdaderas (1, m)
    
    Returns:
        cost: valor del costo
    """
    m = Y.shape[1]
    cost = -(1/m) * np.sum(Y*np.log(A3+1e-9) + (1-Y)*np.log(1-A3+1e-9))
    return cost

def backward(X, Y, Z1, A1, Z2, A2, Z3, A3, W1, W2, W3):
    """
    Retropropagación (3 capas)
    
    Args:
        X: datos de entrada (n_x, m)
        Y: etiquetas verdaderas (1, m)
        Z1, A1, Z2, A2, Z3, A3: valores intermedios del forward
        W1, W2, W3: pesos
    
    Returns:
        dW1, db1, dW2, db2, dW3, db3: gradientes
    """
    m = X.shape[1]

    dZ3 = A3 - Y
    dW3 = (1/m) * dZ3.dot(A2.T)
    db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = W3.T.dot(dZ3) * relu_der(Z2)
    dW2 = (1/m) * dZ2.dot(A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * relu_der(Z1)
    dW1 = (1/m) * dZ1.dot(X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3

def update(W1, b1, W2, b2, W3, b3,
           dW1, db1, dW2, db2, dW3, db3, lr=0.01):
    """
    Actualiza parámetros usando gradient descent
    
    Args:
        W1, b1, W2, b2, W3, b3: parámetros actuales
        dW1, db1, dW2, db2, dW3, db3: gradientes
        lr: learning rate
    
    Returns:
        W1, b1, W2, b2, W3, b3: parámetros actualizados
    """
    W1 -= lr*dW1
    b1 -= lr*db1
    W2 -= lr*dW2
    b2 -= lr*db2
    W3 -= lr*dW3
    b3 -= lr*db3
    return W1, b1, W2, b2, W3, b3