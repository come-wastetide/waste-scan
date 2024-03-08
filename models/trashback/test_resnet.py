import sklearn.metrics as metrics
import torch




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

    # Afficher la matrice de confusion
    print(confusion_matrix)