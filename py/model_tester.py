from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

class ModelTester:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_and_evaluate_model(self, model, model_name):
        """Entrena y evalúa un modelo."""
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_pred)

        print(f"Resultados para {model_name}:")
        print(f"  Precisión: {accuracy:.2f}, F1-Score: {f1:.2f}, AUC: {auc:.2f}")

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