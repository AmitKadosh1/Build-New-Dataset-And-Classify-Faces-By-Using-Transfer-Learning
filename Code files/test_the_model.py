import torch
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
from build_and_train_the_model import reference_model, calculate_accuracy, initialize_model

#Relevant links:
#https://discuss.pytorch.org/t/error-loading-saved-model/8371/5
#https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
#https://pytorch.org/tutorials/beginner/saving_loading_models.html

def calculate_confusion_matrix(model, dataloader, device, num_classes):
    model.eval() #Moving the model to evaluation mode (affects the normalization and dropout layers)
    confusion_matrix = np.zeros([num_classes,num_classes],int)
    with torch.no_grad(): #During the accuracy calculation do not change the gradients
        for images,labels in dataloader: #Go over all the images and labels in the batch
            images = images.to(device) #Transferring the images to the device
            labels = labels.to(device) #Transferring the labels to the device
            output = model(images) #Calculation the otput of the model
            _ ,prediction = torch.max(output.data, 1) #Getting the maximum index from the output vector
            for i, label in enumerate(labels):
                confusion_matrix[label.item(), prediction[i].item()] += 1 #Build the confusion matrix
    return confusion_matrix

project_folder = "\\".join(os.path.dirname(os.path.realpath(__file__)).split("\\")[0:-1])
#Building a path to the main folder where the models will be saved
models_and_results_folder = os.path.join(project_folder, "Models and Results")
nums_models_in_folder = [int(d) for d in os.listdir(models_and_results_folder)]
#The maximum model number is the last built model.
num_model = max(nums_models_in_folder) if (len(nums_models_in_folder) != 0) else 0
#num_model = <number> #You can also choose specific model number to test
save_folder = os.path.join(models_and_results_folder, str(num_model))
test_ds_path = os.path.join(project_folder, "Dataset Folders", "Final dataset", "test")

#Updating the text file we opened when training the model and adding the results on the test
text_file = open(os.path.join(save_folder, "Results.txt"), 'r+')
lines = [line for line in text_file]
model_name = (lines[0].split(":")[-1])[1:-1] #Get the model name from the text file
feature_extract = True if ((lines[1].split(":")[-1])[1:-1] == 'True') else False #Get feature_extract parameter from the text file
num_classes = len([d for d in os.listdir(test_ds_path)]) #Get number of classes from the text file
test_batch_size = 32 #Define batch size for calcuale the accuracy on the test set
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Define the device
best_model = initialize_model(model_name, num_classes, feature_extract).to(device) #Create the model
best_model.load_state_dict(torch.load(os.path.join(save_folder, "best_model.pth"))) #Load the whight for the model from best model file
#Applying transformations to the test set
transform_test = transforms.Compose(
    [transforms.ToTensor()]) if model_name == "our_model" else transforms.Compose(
    [transforms.Resize(224), transforms.ToTensor()])

test_ds = datasets.ImageFolder(test_ds_path, transform=transform_test) #Loading the test dataset from the test images folder
class_names_list = [d for d in os.listdir(test_ds_path) if os.path.isdir(os.path.join(test_ds_path, d))]
test_dataloader = DataLoader(test_ds, batch_size=test_batch_size)
#Writing the result on the test set into the test file
text_file.write(f"Best model: Test accuracy = {calculate_accuracy(best_model, test_dataloader, device):.4f}%")

confusion_matrix = calculate_confusion_matrix(best_model, test_dataloader, device, num_classes)
fig = plt.figure(figsize = (10,7))
df_cm = pd.DataFrame(confusion_matrix, index=class_names_list, columns=class_names_list)
seaborn.heatmap(df_cm, annot=True) #Display the confusion matrix
plt.show()
fig.savefig(os.path.join(save_folder, "confusion matrix.png")) #Save the confusion matrix to file
text_file.close() #Close the text file