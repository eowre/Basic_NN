import numpy as np

def sigmoid(x):
    """Sigmoid activation function: squashes input to [0,1] range"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid: used in backpropagation"""
    return sigmoid(x) * (1 - sigmoid(x))

# ===============================
# Parameters
# ===============================

# Set a seed for reproducibility 
np.random.seed(36)

# Define layer sizes
input_size = 2
hidden_size = 3
output_size = 1

# Random initilizations of weights and biases
weight1 = np.random.randn(input_size, hidden_size) # (2,2)
bias1 = np.zeros((1, hidden_size)) # (1,2)

weight2 = np.random.randn(hidden_size, output_size) # (2, 1)
bias2 = np.zeros((1, output_size)) # (1, 1)

# ===============================
# Forward Propagation
# ===============================

def forward(x):
    """
    Perform forward propagation through the network
    Returns intermediate values for use in backprop
    """
    z1 = np.dot(x, weight1) + bias1 # Linear transform for hidden layer
    a1 = sigmoid(z1) # Activation for hidden layer

    z2 = np.dot(a1, weight2) + bias2 # Linear transform for output layer
    a2 = sigmoid(z2) # Final output activation

    return z1, a1, z2, a2

# ===============================
# Loss Function
# ===============================

def compute_loss(y_true, y_pred):
    """"Binary cross-entropy loss function"""
    m = y_true.shape[0]
    # Add small epsilon to avoid log(0)
    return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))

# ===============================
# Backward Propagation
# ===============================

def backward(x, y, z1, a1, z2, a2, learning_rate):
    """Compute gradients and update weights and biases"""
    global weight1, bias1, weight2, bias2

    m = x.shape[0]

    # Gradient of loss w.r.t A2 (output error)
    delta_z2 = a2 - y    # (m, 1)
    delta_weight2 = np.dot(a1.T, delta_z2) / m
    delta_bias2 = np.sum(delta_z2, axis=0, keepdims=True) / m

    # Gradient of loss w.r.t A1 (hidden layer error)
    delta_z1 = np.dot(delta_z2, weight2.T) * sigmoid_derivative(z1)
    delta_weight1 = np.dot(x.T, delta_z1) / m
    delta_bias1 = np.sum(delta_z1, axis=0, keepdims=True) / m

    # Update weights and biases (gradient descent step)
    weight1 -= learning_rate * delta_weight1
    bias1 -= learning_rate * delta_bias1
    weight2 -= learning_rate * delta_weight2
    bias2 -= learning_rate * delta_bias2

# ===============================
# Training Loop
# ===============================
def train(x, y, epochs, learning_rate):
    """Train the neural network using backpropagation"""
    for epoch in range(epochs):

        # Forward Pass
        z1, a1, z2, a2 = forward(x)
        loss = compute_loss(y, a2)

        # Backward pass 
        backward(x, y, z1, a1, z2, a2, learning_rate)

        # Print loss every 1000 epochs
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

# ===============================
# Prediction Function
# ===============================

def predict(x):
    """Make predictions using the trained network"""
    _, _, _, a2 = forward(x)
    return (a2 > 0.5).astype(int) # Convert probablities to 0/1

# ===============================
# Test the Model
# ===============================