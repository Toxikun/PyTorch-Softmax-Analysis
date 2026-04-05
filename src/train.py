import torch

def compute_regularization_loss(model, reg_type, lambda_val, alpha=0.5):
    #Compute the regularization penalty term.
    #regtype can be ridge, lasso, or elasticnet
    W = model.W
    
    if reg_type == 'ridge':
        #L2 Penalty
        loss = lambda_val * torch.sum(W ** 2)
    elif reg_type == 'lasso':
        #L1 Penalty
        loss = lambda_val * torch.sum(torch.abs(W))
    elif reg_type == 'elasticnet':
        #Combination of L1 and L2
        loss = lambda_val * (alpha * torch.sum(torch.abs(W)) + (1 - alpha) * torch.sum(W ** 2))
    else:
        loss = torch.tensor(0.0)
        
    return loss

def train_model(model, X_train, y_train, X_val, y_val, X_test, y_test, lr, reg_type, epochs=50):#trains the model
    #CrossEntropyLoss computes softmax and negative log likelihood
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    #Step scheduler: reduces LR by half every 20 epochs(this was optional in the pdf so I added
    #to prevent overfitting)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    history = {'train_loss': [], 'val_loss': [], 'test_loss': []}
    
    lambda_val = 0.01 #You can tune this
    
    for epoch in range(epochs):
        #Forward pass
        logits = model(X_train)
        base_loss = criterion(logits, y_train)
        
        #Add regularization penalty
        reg_loss = compute_regularization_loss(model, reg_type, lambda_val)
        total_loss = base_loss + reg_loss
        
        #Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        #Step the learning rate scheduler
        scheduler.step()
        
        #Evaluate on train
        history['train_loss'].append(total_loss.item())
        
        #Evaluate on validation
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val) + compute_regularization_loss(model, reg_type, lambda_val)
            history['val_loss'].append(val_loss.item())
            
            test_logits = model(X_test)
            test_loss = criterion(test_logits, y_test) + compute_regularization_loss(model, reg_type, lambda_val)
            history['test_loss'].append(test_loss.item())
            
    return history

def k_fold_cross_validation(X, y, k=3):#Implement k-fold cross validation using sklearn KFold or manually.
    #Evaluate linear, quadratic, and 3rd order models.
    from sklearn.model_selection import KFold
    from model import CustomSoftmaxRegression
    from dataset import expand_features
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)#creates the k-fold cross validation
    degrees = [1, 2, 3]#sets the degrees to 1, 2, and 3
    best_degree = 1#sets the best degree to 1
    best_avg_loss = float('inf')#sets the best average loss to infinity
    
    lr = 0.1 #Default learning rate for tuning
    num_classes = len(torch.unique(y))#gets the number of classes
    
    for degree in degrees:#iterates through the degrees
        X_exp = expand_features(X, degree=degree)#expands the features
        num_features = X_exp.shape[1]#gets the number of features
        
        fold_val_losses = []#creates a list to store the validation losses
        
        for train_idx, val_idx in kf.split(X_exp):#iterates through the k-fold cross validation
            X_fold_train, X_fold_val = X_exp[train_idx], X_exp[val_idx]#splits the data into training and validation sets
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]#splits the data into training and validation sets
            
            model = CustomSoftmaxRegression(num_features, num_classes)#initializes the model
            
            #Evaluate using default Ridge to stay stable
            history = train_model(model, X_fold_train, y_fold_train, X_fold_val, y_fold_val, X_fold_val, y_fold_val, lr=lr, reg_type='ridge', epochs=50)
            
            final_val_loss = history['val_loss'][-1]#gets the final validation loss
            fold_val_losses.append(final_val_loss)#adds the final validation loss to the list
            
        avg_val_loss = sum(fold_val_losses) / k#computes the average validation loss
        print(f"Degree {degree} CV Validation Loss: {avg_val_loss:.4f}")#prints the average validation loss
        
        if avg_val_loss < best_avg_loss:
            best_avg_loss = avg_val_loss
            best_degree = degree
            
    print(f"Best degree selected: {best_degree}")
    return best_degree
