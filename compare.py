from sklearn.linear_model import LinearRegression

from dataset import create_dataset
from model import predict

# My learned parameters
m = 7.8681
b = 31.3711

X, Y = create_dataset()

# My model predictions
my_predictions = predict(X, m, b)

# sklearn model
X_train = X.reshape(-1, 1)

model = LinearRegression()
model.fit(X_train, Y)

sk_predictions = model.predict(X_train)

print("X\tMy Prediction\tSklearn Prediction")
print("-" * 45)

for x, mine, skl in zip(X, my_predictions, sk_predictions):
    print(f"{x:.0f}\t{mine:.2f}\t\t{skl:.2f}")