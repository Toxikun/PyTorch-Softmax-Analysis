from dataset import get_iris_data, expand_features
from model import CustomSoftmaxRegression
from train import train_model, k_fold_cross_validation
from plot import plot_loss_curves, compute_metrics
import torch

def main():
    # 1. Load Data
    X_train, y_train, X_val, y_val, X_test, y_test = get_iris_data()
    
    # 2. Find the best polynomial feature degree using cross-validation (k=3)
    print("Performing 3-fold CV to determine the best model degree...")
    best_degree = k_fold_cross_validation(X_train, y_train, k=3)
    
    # Expand features using the best degree
    X_train_exp = expand_features(X_train, degree=best_degree)
    X_val_exp = expand_features(X_val, degree=best_degree)
    X_test_exp = expand_features(X_test, degree=best_degree)
    
    num_features = X_train_exp.shape[1]
    num_classes = 3 # Iris has 3 classes
    
    # 3. Choose learning rates
    # 0.00001 <= l1 <= 0.00002
    # 0.001 <= l2 <= 0.002
    # 0.1 <= l3 <= 0.2
    learning_rates = {
        'l1': 0.000015,
        'l2': 0.0015,
        'l3': 0.15
    }
    
    regularizations = ['ridge', 'lasso', 'elasticnet']
    
    # 4. Train 9 models
    epochs = 50
    results_dict = {}
    
    for reg in regularizations:
        for lr_name, lr in learning_rates.items():
            model_name = f"{reg}_{lr_name}"
            print(f"Training model: {model_name} (reg={reg}, lr={lr})")
            
            # Re-initialize the model for each training explicitly
            model = CustomSoftmaxRegression(num_features, num_classes)
            
            history = train_model(
                model=model, 
                X_train=X_train_exp, y_train=y_train, 
                X_val=X_val_exp, y_val=y_val, 
                X_test=X_test_exp, y_test=y_test, 
                lr=lr, reg_type=reg, epochs=epochs
            )
            
            results_dict[model_name] = history
            results_dict[model_name]['model'] = model
    
    # 5. Plot Loss Curves
    plot_loss_curves(results_dict, epochs=epochs)
    
    # 6. Evaluate lowest validation error model
    best_model_name = min(results_dict, key=lambda k: results_dict[k]['val_loss'][-1])
    print(f"\nBest model based on validation loss: {best_model_name}")
    
    best_model = results_dict[best_model_name]['model']
    
    with torch.no_grad():
        test_logits = best_model(X_test_exp)
        y_pred_test = test_logits.argmax(dim=1)
        
    acc, prec, rec, f1 = compute_metrics(y_test, y_pred_test)
    print(f"Test Set Metrics for {best_model_name}:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

if __name__ == "__main__":
    main()
