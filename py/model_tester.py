from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import mlflow
from mlflow.sklearn import log_model

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import mlflow
from mlflow.sklearn import log_model

class ModelTester:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_and_evaluate_model(self, model, model_name):
        with mlflow.start_run():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

            # Si el modelo tiene un método predict_proba, úsalo para calcular el AUC
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(self.X_test)[:, 1]  # Asume que la segunda columna es la probabilidad de la clase positiva
                auc = roc_auc_score(self.y_test, y_score)
            else:
                auc = "No disponible"  # O maneja de otra manera si el modelo no soporta predict_proba

            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, pos_label='Y')  # Asegúrate de que 'Y' es tu etiqueta positiva

            print(f"Resultados para {model_name}:")
            print(f"  Precisión: {accuracy:.2f}, F1-Score: {f1:.2f}, AUC: {auc}")

            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            if auc != "No disponible":
                mlflow.log_metric("auc", auc)

            # Ruta para guardar el modelo
            model_path = "model"
            mlflow.sklearn.log_model(model, artifact_path=model_path, registered_model_name=model_name)


    def grid_search(self, model, param_grid, model_name):
        """Realiza una búsqueda en cuadrícula para encontrar los mejores hiperparámetros."""
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_

        print(f"\nMejores hiperparámetros para {model_name}: {grid_search.best_params_}")
        self.train_and_evaluate_model(best_model, f"{model_name} (Mejor)")

    def ensemble_models(self):
        """Crea y retorna un ensamblaje de varios modelos."""
        print("\nProbando Ensamblaje de Modelos...")
        logistic = LogisticRegression()
        random_forest = RandomForestClassifier()
        svm = SVC(probability=True)
        adaboost = AdaBoostClassifier()

        ensemble_model = VotingClassifier(estimators=[
            ('lr', logistic),
            ('rf', random_forest),
            ('svc', svm),
            ('ada', adaboost)
        ], voting='soft')

        self.train_and_evaluate_model(ensemble_model, "Ensamblaje de Modelos")
        return ensemble_model