import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn styling for all matplotlib plots
sns.set_theme(style="darkgrid", palette="tab10")

def plot_loss_curves(results_dict, epochs=50):
    """
    Draw train loss, validation loss, and test loss vs epoch figures.
    results_dict should contain 9 models data, e.g.:
    results_dict['ridge_l1'] = {'train': [...], 'val': [...], 'test': [...]}
    Each figure will contain 9 lines.
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    model_names = list(results_dict.keys())
    
    # Plot Train Loss
    for name in model_names:
        axes[0].plot(range(epochs), results_dict[name]['train_loss'], label=name)
    axes[0].set_title('Train Loss vs Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # Plot Validation Loss
    for name in model_names:
        axes[1].plot(range(epochs), results_dict[name]['val_loss'], label=name)
    axes[1].set_title('Validation Loss vs Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    
    # Plot Test Loss
    for name in model_names:
        axes[2].plot(range(epochs), results_dict[name]['test_loss'], label=name)
    axes[2].set_title('Test Loss vs Epoch')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('loss_figures.png')
    plt.show()

def compute_metrics(y_true, y_pred):
    """
    Computes accuracy, precision, recall and f1-score.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return acc, prec, rec, f1
