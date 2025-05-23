import numpy as np
import basic_nn_XOR

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

epochs = 10000
batch_size = 2
learning_rate = 0.1

def mini_batch_train(x, y, epochs, batch_size, learning_rate):
    """Train the neural network using mini-batch gradient descent"""

    for epoch in range(epochs):
        # Shuffle the data
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        # Mini-batch training
        epoch_loss = 0
        for start in range(0, x.shape[0], batch_size):
            end = start + batch_size
            x_batch = x_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Forward Pass
            z1, a1, z2, a2 = basic_nn_XOR.forward(x_batch)

            # loss calculation
            batch_loss = basic_nn_XOR.compute_loss(y_batch, a2)
            epoch_loss += batch_loss * x_batch.shape[0] # weight by batch size

            # Backward Pass
            basic_nn_XOR.backward(x_batch, y_batch, z1, a1, z2, a2, learning_rate)

        epoch_loss /= x.shape[0]  # Average loss over the entire dataset
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

def predict(x):
    """Predict using the trained model"""
    _, _, _, a2 = basic_nn_XOR.forward(x)
    return np.round(a2).astype(int)  # Round to get binary output
