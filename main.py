from dataset import create_dataset
from model import predict
from loss import mse
from gradient import compute_gradients

# Load dataset
X, Y = create_dataset()

# Initial parameters
m = 5.0
b = 30.0

learning_rate = 0.01

# Training settings
max_epochs = 1000
tolerance =  1e-2

previous_loss = float("inf")

print("Epoch\tm\t\tb\t\tLoss")
print("-" * 60)

for epoch in range(max_epochs):

    # Make predictions
    predictions = predict(X, m, b)

    # Calculate loss
    loss = mse(Y, predictions)

    # Stop if loss is no longer improving
    if abs(previous_loss - loss) < tolerance:
        print(f"\nTraining converged after {epoch} epochs.")
        break

    # Compute gradients
    gradient_m, gradient_b = compute_gradients(X, Y, m, b)

    # Update parameters
    m = m - learning_rate * gradient_m
    b = b - learning_rate * gradient_b

    # Store current loss
    previous_loss = loss

    # Print progress
    print(f"{epoch + 1}\t{m:.4f}\t\t{b:.4f}\t\t{loss:.6f}")

print("\nModel Training Finished")
print("-----------------------")
print(f"Final m = {m:.4f}")
print(f"Final b = {b:.4f}")
print(f"Final Loss = {loss:.6f}")