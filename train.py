from database import connect_to_db, close_connection, insert_training_run, insert_metrics, insert_predictions
from etl import process_pipeline
from model_sklearn import SklearnModel
from model_pytorch import PyTorchModel
from schemas import SKLEARN_PARAMS, PYTORCH_PARAMS
from sklearn.model_selection import train_test_split
from utils import get_class_balance
import uuid
import json
import joblib
import torch
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

#Training functions for different model types


#Train Sklearn Model
def train_sklearn_model(connection, X_train, y_train, X_test, y_test, hyperparameters):
    print('*'*50)
    print("Training Sklearn Model")

    #generate unique run id, create model, train, evaluate, store results
    run_id=str(uuid.uuid4())
    model=SklearnModel(**hyperparameters)
    model.train(X_train, y_train)
    metrics=model.evaluate(X_test, y_test)
    predictions, probabilities=model.predict(X_test)
    predictions_dict={}

    #calculate additional metrics
    for i, record_id in enumerate(X_test.index):
        predictions_dict[record_id]={
            'predicted_label': int(predictions[i]),
            'predicted_probability': float(probabilities[i])
        }
    #calculate standard metrics
    insert_training_run(
        connection,
        run_id,
        model_name='RandomForestClassifier',
        algorithm='sklearn',
        hyperparameters=json.dumps(hyperparameters),
        train_rows=len(X_train),
        test_rows=len(X_test)
    )
    #insert metrics and predictions
    insert_metrics(connection, run_id, metrics)
    insert_predictions(connection, run_id, predictions_dict)

    model_path=f'models/sklearn_model_{run_id[:8]}.pkl'
    model.save_model(model_path)
    print(f"Sklearn model training complete. Run ID: {run_id}")
    print('*'*50)
    return run_id, model, metrics, predictions_dict

def train_pytorch_model(connection, X_train, y_train, X_test, y_test, hyperparameters):
    print('*'*50)
    print("Training PyTorch Model")
    run_id=str(uuid.uuid4())

    #convert to numpy arrays from pandas
    X_train_np=X_train.values if hasattr(X_train,'values') else X_train
    y_train_np=y_train.values if hasattr(y_train,'values') else y_train
    X_test_np=X_test.values if hasattr(X_test,'values') else X_test
    y_test_np=y_test.values if hasattr(y_test,'values') else y_test


    #create model, train, evaluate, store results
    input_dim=X_train_np.shape[1]
    model=PyTorchModel(input_dim=input_dim, **hyperparameters)

    X_train_split,X_val,y_train_split,y_val=train_test_split(X_train_np,y_train_np,test_size=0.2,random_state=42)
    history= model.train(X_train_split,y_train_split,X_val,y_val)

    metrics=model.evaluate(X_test_np,y_test_np)
    
    print("Test Metrics:")
    for name,value in metrics.items():
        print(f"{name}: {value}")
    
    #get predictions
    predictions, probabilities=model.predict(X_test_np)
    x_true= y_test_np
    y_pred= predictions
    y_proba= probabilities
    #calculate standard metrics
    metrics={
        'accuracy': accuracy_score(x_true, y_pred),
        'precision': precision_score(x_true, y_pred, zero_division=0),
        'recall': recall_score(x_true, y_pred, zero_division=0),
        'f1_score': f1_score(x_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(x_true, y_proba) if len(set(x_true)) > 1 else None
    }
    #store predictions
    predictions_dict={}
    for i, record_id in enumerate(X_test.index):
        predictions_dict[record_id]={
            'predicted_label': int(predictions[i]),
            'predicted_probability': float(probabilities[i])
        }
    insert_training_run(
        connection,
        run_id,
        model_name='PyTorch Nueral Network',
        algorithm='pytorch',
        hyperparameters=json.dumps(hyperparameters),
        train_rows=len(X_train),
        test_rows=len(X_test)
    )
    insert_metrics(connection, run_id, metrics)
    insert_predictions(connection, run_id, predictions_dict)

    model_path=f'models/pytorch_model_{run_id[:8]}.pt'
    model.save_model(model_path)
    print(f"PyTorch model training complete. Run ID: {run_id}")
    print('*'*50)
    return run_id, model, metrics, predictions_dict



#Training Pipeline
def main():
    print('*'*50)
    print("Starting training pipeline")
    connection=connect_to_db()

    try:
        #Calling ETL process to get training and test data
        print("ETL Processing")
        print("Processing training data")
        X_train, y_train, encoder, scaler=process_pipeline(connection, split='train', fit=True)
        
        print("Processing test data")
        X_test, y_test, _, _=process_pipeline(connection, split='test', fit=False, encoder=encoder, scaler=scaler)
        print(f"Test Data:{X_test.shape[0]} rsamples, {X_test.shape[1]} features")

        train_balance=get_class_balance(y_train)
        test_balance=get_class_balance(y_test)
        print(f"Training set class balance: {train_balance}")
        print(f"Test set class balance: {test_balance}")



        
        #Train Sklearn Model and PyTorch Model

        #Deterministic model-only 1 run
        print("Training Sklearn Model")
        sklearn_run_id, sklearn_model, sklearn_metrics, sklearn_predictions=train_sklearn_model(
            connection,
            X_train,
            y_train,
            X_test,
            y_test,
            SKLEARN_PARAMS
        )
        #Running multiple runs
        runs=10
        for run in range(runs):
            print(f"Run {run+1}/{runs}")
            print("Training PyTorch Model")
            pytorch_run_id, pytorch_model, pytorch_metrics, pytorch_predictions=train_pytorch_model(
                connection,
                X_train,
                y_train,
                X_test,
                y_test,
                PYTORCH_PARAMS
            )

            print("Training pipeline complete")
            print('*'*50)

            #Compare model performances
            print("Comaring model performances:")
            print(f"{'Metric':<15}{'Sklearn':<12}{'PyTorch':<12} {'Winner':<10}")
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
                sklearn_value=sklearn_metrics[metric]
                pytorch_value=pytorch_metrics[metric]
                if sklearn_value>pytorch_value:
                    winner='Sklearn'
                elif pytorch_value>sklearn_value:
                    winner='PyTorch'
                else:
                    winner='Tie'
                print(f"{metric:<15}{sklearn_value:<12.4f}{pytorch_value:<12.4f}{winner:<10}")
            print('*'*50)
  
        print("All training runs complete.")


    except Exception as e:
        print(f"An error occurred during training: {e}")

    finally:
        close_connection(connection)
    

if __name__ == "__main__":
    main()