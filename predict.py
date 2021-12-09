import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

# set up model for inference
model_4class = torch.load('..\\model\\main_model\\model_4class.pt', map_location=torch.device('cpu'))
model_2class = torch.load('..\\model\\main_model\\model_2class.pt', map_location=torch.device('cpu'))

class_dict = {0:'COVID', 1:'Lung_Opacity', 2:'Normal', 3:'Viral Pneumonia'}

def predict_4class(input_image):
    device = torch.device("cpu")
    model_4class.to(device)
    model_4class.eval()

    image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    transform = transforms.ToTensor()

    tensor = transform(image)
    tensor =  torch.unsqueeze(tensor, dim=0)
    tensor = tensor.to(device)

    model_output = model_4class(tensor)

    prediction = np.argmax(model_output.detach().numpy(), axis=1)[0]

    return prediction

def predict_2class(input_image):
    device = torch.device("cpu")
    model_2class.to(device)
    model_2class.eval()

    image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    transform = transforms.ToTensor()

    tensor = transform(image)
    tensor =  torch.unsqueeze(tensor, dim=0)
    tensor = tensor.to(device)

    model_output = model_2class(tensor)

    prediction = np.argmax(model_output.detach().numpy(), axis=1)[0]

    prediction = 2 if prediction == 1 else 0

    return prediction