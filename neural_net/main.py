import numpy as np
import basic_nn_XOR
import mini_batch

def main():
    """Main function to run the XOR neural network example"""
    # Training data for XOR
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Train the neural networks
    basic_nn_XOR.train(x, y, epochs=10000, learning_rate=0.1)
    mini_batch.mini_batch_train(x, y, epochs=10000, batch_size=3, learning_rate=0.1)

    # Test the model
    predictions_basic_nn = basic_nn_XOR.predict(x)
    predictions_mini_batch = mini_batch.predict(x)
    print("Final Predictions:")
    print(predictions_basic_nn)  # Should approximate XOR: [0, 1, 1, 0]
    print("Final Predictions (Mini-Batch):")
    print(predictions_mini_batch)  # Should approximate XOR: [0, 1, 1, 0]


if __name__ == "__main__":

    main()