import sklearn.metrics as metrics
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix




def test(trainer,model,test_loader,ckpt_path='best'):

    trainer.test(model, test_loader, ckpt_path=ckpt_path)


def print_confusion_matrix(path_to_checkpoints):

    ''' 
    To complete ! 
    '''

    # Charger le modèle entraîné
    model = LitResNeXt.load_from_checkpoint(path_to_checkpoints)

    # Désactiver le mode d'entraînement
    model.eval()


    # Prédire les étiquettes pour les données de test
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in test_loader:
            images, targets = batch
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            predictions.extend(predicted.cpu().numpy())
            labels.extend(targets.cpu().numpy())

    # Calculer la matrice de confusion
    confusion_matrix = metrics.confusion_matrix(labels, predictions)

    
    # Define the class labels
    class_names = train_dataset.classes

    # Define the confusion matrix
    cm = np.array(confusion_matrix)

    # Normalize the confusion matrix to percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Set the labels for the rows and columns
    ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label')

    # Set the threshold for different colors
    threshold = cm_normalized.max() / 2.

    # Add the values to the cells
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, f'{cm[i, j]} ({cm_normalized[i, j]:.1f}%)',
                horizontalalignment="center",
                color="white" if cm_normalized[i, j] > threshold else "black")

    # Add a title
    plt.title('Confusion Matrix')

    # Show the plot
    plt.show()

    # Afficher la matrice de confusion
    print(confusion_matrix)