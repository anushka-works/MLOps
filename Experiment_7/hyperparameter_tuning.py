import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("Hyperparameter_Tuning")

models = [
    ("GridSearch_RF", GridSearchCV(RandomForestClassifier(), 
     {"n_estimators": [20, 50]}, cv=3)),
    ("RandomSearch_SVC", RandomizedSearchCV(SVC(), 
     {"C": [0.1, 1, 10]}, n_iter=3, cv=3))
]

for name, model in models:
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        mlflow.log_params(model.best_params_)
        mlflow.log_metric("accuracy", acc)
        print(name, model.best_params_, "Accuracy:", acc)

print("Tuning Completed!")