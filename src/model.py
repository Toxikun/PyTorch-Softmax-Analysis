import torch

class CustomSoftmaxRegression:
    def __init__(self, num_features, num_classes):#initializes the weights and biases
        self.W = (torch.randn(num_features, num_classes) * 0.01).requires_grad_(True)
        self.b = (torch.randn(num_classes) * 0.01).requires_grad_(True)
        
    def parameters(self):#returns the weights and biases

        return [self.W, self.b]

    def forward(self, X):#computes the forward pass

        logits = torch.matmul(X, self.W) + self.b
        return logits
    
    def __call__(self, X):#calls the forward pass
        return self.forward(X)
