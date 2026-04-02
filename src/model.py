import torch

class CustomSoftmaxRegression:
    def __init__(self, num_features, num_classes):
        """
        Initialize weights and biases for Softmax Regression manually as per assignment.
        Do not use nn.Linear.
        """
        # Define weights and bias as trainable parameters (requires_grad=True)
        # Weights shape should be (num_features, num_classes)
        # Bias shape should be (num_classes,)
        # Scale initialization by 0.01 to prevent large initial softmax losses
        self.W = (torch.randn(num_features, num_classes) * 0.01).requires_grad_(True)
        self.b = (torch.randn(num_classes) * 0.01).requires_grad_(True)
        
    def parameters(self):
        """
        Return the list of trainable parameters.
        """
        return [self.W, self.b]

    def forward(self, X):
        """
        Forward pass using matrix operations.
        X shape: (num_samples, num_features)
        Compute Z = X @ W + b
        Apply softmax to compute probabilities (if needed) or return raw logits.
        PyTorch's CrossEntropyLoss expects raw logits.
        """
        logits = torch.matmul(X, self.W) + self.b
        return logits
    
    def __call__(self, X):
        return self.forward(X)
