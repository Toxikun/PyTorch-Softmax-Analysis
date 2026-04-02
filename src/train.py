import torch

def compute_regularization_loss(model, reg_type, lambda_val, alpha=0.5):
    """
    Compute the regularization penalty term.
    reg_type can be 'ridge', 'lasso', or 'elasticnet'.
    """
    W = model.W
    
    if reg_type == 'ridge':
        # L2 Penalty
        loss = lambda_val * torch.sum(W ** 2)
    elif reg_type == 'lasso':
        # L1 Penalty
        loss = lambda_val * torch.sum(torch.abs(W))
    elif reg_type == 'elasticnet':
        # Combination of L1 and L2
        loss = lambda_val * (alpha * torch.sum(torch.abs(W)) + (1 - alpha) * torch.sum(W ** 2))
    else:
        loss = torch.tensor(0.0)
        
    return loss

def train_model(model, X_train, y_train, X_val, y_val, X_test, y_test, lr, reg_type, epochs=50):
    """
    Train a given model from scratch for a fixed number of epochs.
    Return the history of train/val/test losses to be able to plot later.
    """
    # CrossEntropyLoss computes softmax and negative log likelihood
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'test_loss': []}
    
    lambda_val = 0.01 # You can tune this
    
    for epoch in range(epochs):
        # Forward pass
        logits = model(X_train)
        base_loss = criterion(logits, y_train)
        
        # Add regularization penalty
        reg_loss = compute_regularization_loss(model, reg_type, lambda_val)
        total_loss = base_loss + reg_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Evaluate on train
        history['train_loss'].append(total_loss.item())
        
        # Evaluate on validation
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val) + compute_regularization_loss(model, reg_type, lambda_val)
            history['val_loss'].append(val_loss.item())
            
            test_logits = model(X_test)
            test_loss = criterion(test_logits, y_test) + compute_regularization_loss(model, reg_type, lambda_val)
            history['test_loss'].append(test_loss.item())
            
    return history

def k_fold_cross_validation(X, y, k=3):
    """
    Implement k-fold cross validation using sklearn KFold or manually.
    Evaluate linear, quadratic, and 3rd order models.
    """
    from sklearn.model_selection import KFold
    from model import CustomSoftmaxRegression
    from dataset import expand_features
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    degrees = [1, 2, 3]
    best_degree = 1
    best_avg_loss = float('inf')
    
    lr = 0.1 # Default learning rate for tuning
    num_classes = len(torch.unique(y))
    
    for degree in degrees:
        X_exp = expand_features(X, degree=degree)
        num_features = X_exp.shape[1]
        
        fold_val_losses = []
        
        for train_idx, val_idx in kf.split(X_exp):
            X_fold_train, X_fold_val = X_exp[train_idx], X_exp[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            model = CustomSoftmaxRegression(num_features, num_classes)
            
            # Evaluate using default Ridge to stay stable
            history = train_model(model, X_fold_train, y_fold_train, X_fold_val, y_fold_val, X_fold_val, y_fold_val, lr=lr, reg_type='ridge', epochs=50)
            
            final_val_loss = history['val_loss'][-1]
            fold_val_losses.append(final_val_loss)
            
        avg_val_loss = sum(fold_val_losses) / k
        print(f"Degree {degree} CV Validation Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_avg_loss:
            best_avg_loss = avg_val_loss
            best_degree = degree
            
    print(f"Best degree selected: {best_degree}")
    return best_degree
