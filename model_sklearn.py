from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# Define a class for the RandomForest model
class SklearnModel:
    def __init__(self, **hyperparameters):
        self.model= RandomForestClassifier(**hyperparameters)
        print("Initialized RandomForestClassifier with hyperparameters:", hyperparameters)
    #train method to fit the model
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print(f"Trained model on {len(X_train)} records")
    
    #predict method to generate predictions
    def predict(self, X_test):
        predictions=self.model.predict(X_test)
        probabilities=self.model.predict_proba(X_test)[:, 1]
        print(f"Generated predictions for {len(X_test)} records")
        return predictions, probabilities
    #evaluate method to assess model performance
    def evaluate(self, X_test, y_test):
        predictions, probabilities = self.predict(X_test)
        #Calculate evaluation metrics
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1_score': f1_score(y_test, predictions),
            'roc_auc': roc_auc_score(y_test, probabilities)
        }
        print("Evaluated model")
        return metrics
    
    #save and load methods for model  
    def save_model(self, filepath):
       joblib.dump(self.model, filepath)
       print(f"Model saved to {filepath}")
    def load_model(self, filepath):
        self.model=joblib.load(filepath)
        print(f"Model loaded from {filepath}")