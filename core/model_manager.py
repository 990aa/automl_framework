from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

class ModelManager:
    def __init__(self):
        self.models = {
            "RandomForestClassifier": RandomForestClassifier,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "LogisticRegression": LogisticRegression,
            "SVC": SVC,
            "KNeighborsClassifier": KNeighborsClassifier,
            "DecisionTreeClassifier": DecisionTreeClassifier,
        }

    def get_model(self, name):
        model = self.models.get(name)
        if model:
            return model()
        return None

    def list_models(self):
        return list(self.models.keys())