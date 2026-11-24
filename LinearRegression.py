from sklearn.linear_model import LinearRegression

# Input data
X = [[1], [2], [3], [4], [5]]

# Output data
y = [20, 30, 40, 50, 60]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict marks for 6 study hours
prediction = model.predict([[6]])

print("Predicted Marks:", prediction[0])
