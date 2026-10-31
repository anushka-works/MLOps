import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dataset
X, y = load_iris(return_X_y=True)

# Set experiment
mlflow.set_experiment("Iris_Experiment")

# Train model
with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(X, y)

    accuracy = model.score(X, y)

    # Log details
    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("accuracy", accuracy)

    # Save model
    mlflow.sklearn.log_model(
    model,
    name="model",
    skops_trusted_types=["sklearn.tree._tree.Tree"],
    registered_model_name="IrisModel"
)

    print("Accuracy:", accuracy)
    print("Model trained successfully!")