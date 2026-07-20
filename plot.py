import matplotlib.pyplot as plt

m_values = [5, 6, 7, 8, 9, 10]
mse_values = [116.6, 56.4, 18.2, 2.0, 7.8, 35.6]

plt.plot(m_values, mse_values, marker='o')

plt.xlabel("Slope (m)")
plt.ylabel("Mean Squared Error")
plt.title("MSE vs Slope")

plt.grid(True)
plt.show()