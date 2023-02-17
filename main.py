import numpy as np
from linear_regression import train, predict

X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

# Try to Train the system with a range of learning rate (1-0.00001)
w, b = train(X, Y, iterations=100000, lr=0.01)
print("\nw=%.3f, b=%.3f" % (w, b))

# Predict the number of pizzas
print("Prediction: x=%d => y=[%.2f]" % (20, predict(20, w, b)))
