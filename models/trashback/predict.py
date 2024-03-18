from PIL import Image
import torch
from test_resnet import LigResNet,LigResNext

from torchvision import transforms



def get_model_LigResNext(path_to_checkpoints):
    model = LigResNext.load_from_checkpoint(path_to_checkpoints)
    model.eval()
    return model

def get_model_LigResNet(path_to_checkpoints):
    model = LigResNet.load_from_checkpoint(path_to_checkpoints)
    model.eval()
    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_image_class(image_path, model, transform):


    """
    Perform inference on a single image using the specified model and transformations.

    Args:
        image_path (str): Path to the input image.
        model: Pre-trained PyTorch model.
        transform: PyTorch image transformation pipeline.

    Returns:
        predicted_class (int): Predicted class index.
        probability (float): Probability of the predicted class.
    """
    # Load and preprocess the input image
    model.eval()
    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Ensure input tensor and model are on the same device
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)

    # Interpret the model's output
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()

    return predicted_class, probabilities[predicted_class].item()


def predict_test(model,image):

    size = 620,620
    
    data_transforms = transforms.Compose([
        transforms.Resize(size, antialias=None),
        transforms.ToTensor(),
        transforms.Normalize(([0.485, 0.456, 0.406]), ([0.229, 0.224, 0.225]))  # moyennes et écarts-types pour ImageNet
    ])

    # Appliquer les transformations à l'image
    #image_tensor = data_transforms(image)
    
    image_tensor=image

    # Ajouter une dimension batch à l'image
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)
    
    
    
        # Faire une prédiction
    with torch.no_grad():
        output = model(image_tensor)

    # Obtenir la classe prédite (indice de la plus grande valeur de sortie)
    predicted_class = torch.argmax(output, dim=1)
    #label.to(device)
    return predicted_class[0].item()

