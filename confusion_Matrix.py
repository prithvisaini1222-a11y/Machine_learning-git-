
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# Actual values
y_actual = [1]*40 + [1]*10 + [0]*5 + [0]*45

# Predicted values
y_predicted = [1]*40 + [0]*10 + [1]*5 + [0]*45

# Confusion Matrix
cm = confusion_matrix(y_actual, y_predicted)

print("Confusion Matrix:")
print(cm)

# Evaluation Metrics
print("Accuracy:", accuracy_score(y_actual, y_predicted))
print("Precision:", precision_score(y_actual, y_predicted))
print("Recall:", recall_score(y_actual, y_predicted))
print("F1 Score:", f1_score(y_actual, y_predicted))

# Complete Classification Report
print("\nClassification Report:")
print(classification_report(y_actual, y_predicted))
