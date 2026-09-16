import math
import random
import pydl
from pydl import *

# ==========================================================
# Neural Network Example
# Learn the cosine function: y = cos(x)
# ==========================================================

# Create a neural network with:
# 1 input → 3 hidden layers of 10 neurons → 1 output
nn = pydl.Neuronal_network(
    [1, 10, 10, 10, 1],
    pydl.Initialization.he
)

# Use ReLU for every hidden layer
nn.set_activation(
    pydl.Activation(
        pydl.ReLU.forward,
        pydl.ReLU.derivative
    )
)

# Replace the output activation with Tanh
# since cos(x) is naturally bounded between -1 and 1.
nn.neurones[-1][0].activation = pydl.Activation(
    pydl.Tanh.forward,
    pydl.Tanh.derivative
)

# ==========================================================
# Training
# ==========================================================

for _ in range(500000):

    # Generate a random angle in degrees
    angle = random.uniform(-360, 360)

    # Expected output: cosine of the angle
    target = [math.cos(math.radians(angle))]

    # Normalize the input to the range [-1, 1]
    x = [angle / 360]

    # Perform one backpropagation step
    nn.back_propagation(x, target, 0.001)

# ==========================================================
# Save the trained model
# ==========================================================

nn.save_json("data.json")

print("Training complete! Model saved to data.json")

# ==========================================================
# Model Evaluation
# ==========================================================

# Generate a test dataset
test_inputs = []
test_targets = []

for angle in range(-360, 361):
    test_inputs.append([angle / 360])
    test_targets.append([math.cos(math.radians(angle))])

# Compute the mean absolute error
error = nn.evaluate(test_inputs, test_targets)

print(f"Mean Absolute Error: {error:.6f}")

# ==========================================================
# Interactive Prediction
# ==========================================================

while True:

    user_input = input("\nEnter an angle in degrees (or 'q' to quit): ")

    if user_input.lower() == "q":
        print("Goodbye!")
        break

    try:
        angle = float(user_input)

        # Normalize the input exactly like during training
        x = [angle / 360]

        # Neural network prediction
        prediction = nn.predict(x)[0]

        # Actual cosine value
        expected = math.cos(math.radians(angle))

        # Calculate the error
        error = abs(prediction - expected)

        print(f"Angle      : {angle}°")
        print(f"Prediction : {prediction:.6f}")
        print(f"Real cos   : {expected:.6f}")
        print(f"Error      : {error:.6f}")

    except ValueError:
        print("Please enter a valid number.")
