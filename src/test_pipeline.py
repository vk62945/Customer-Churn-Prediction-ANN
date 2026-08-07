from train import train_model

from predict import (
    load_trained_model,
    predict_customer
)

from utils import (
    evaluate_model,
    print_metrics,
    plot_confusion_matrix,
    print_classification_report
)

from config import THRESHOLD

# Train the Model
print("="*60)
print("STEP 1 : TRAINING MODEL")
print("="*60)
(model, history, x_test, y_test, scaler, encoder) = train_model()

# Load Saved Model
print("="*60)
print("STEP 2 : LOADING MODEL")
print("="*60)
model = load_trained_model()
print("Model Loaded Successfully")

#Prediction
y_prob = model.predict(
    x_test,
    verbose = 0
).flatten()

y_pred = (y_prob >= THRESHOLD).astype(int)

# Evaluate
metrics = evaluate_model(
    y_test, y_pred, y_prob
)

print_metrics(metrics)

print_classification_report(y_test, y_pred)

plot_confusion_matrix(y_test, y_pred)

# Test Single Prediction
sample = x_test[0]
result = predict_customer(model, sample)
print(result)

print("="*60)
print("END-TO-END PIPELINE PASSED")
print("="*60)
