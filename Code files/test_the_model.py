import torch
import os
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from build_and_train_the_model import our_model, calculate_accuracy, initialize_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd

#https://discuss.pytorch.org/t/error-loading-saved-model/8371/5
#https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
#https://pytorch.org/tutorials/beginner/saving_loading_models.html

def calculate_confusion_matrix(model, dataloader, device, num_classes):
    model.eval()
    confusion_matrix = np.zeros([num_classes,num_classes],int)
    with torch.no_grad():
        for images,labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            _ ,prediction = torch.max(output.data, 1)
            for i, label in enumerate(labels):
                confusion_matrix[label.item(), prediction[i].item()] += 1
    return confusion_matrix


project_folder = "\\".join(os.path.dirname(os.path.realpath(__file__)).split("\\")[0:-1])
models_and_results_folder = os.path.join(project_folder, "Models and Results")
nums_models_in_folder = [int(d) for d in os.listdir(models_and_results_folder)]
num_model = max(nums_models_in_folder) if (len(nums_models_in_folder) != 0) else 0
#num_model = <number> #You can also choose specific model number to test
save_folder = os.path.join(models_and_results_folder, str(num_model))
test_ds_path = os.path.join(project_folder, "Dataset Folders", "Final dataset", "test")

text_file = open(os.path.join(save_folder, "Results.txt"), 'r+')
lines = [line for line in text_file]
model_name = (lines[0].split(":")[-1])[1:-1]
feature_extract = True if ((lines[1].split(":")[-1])[1:-1] == 'True') else False
num_classes = len([d for d in os.listdir(test_ds_path)])
test_batch_size = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
best_model = initialize_model(model_name, num_classes, feature_extract).to(device)
best_model.load_state_dict(torch.load(os.path.join(save_folder, "best_model.pth")))

transform_test = transforms.Compose(
    [transforms.ToTensor()]) if model_name == "our_model" else transforms.Compose(
    [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

test_ds = datasets.ImageFolder(test_ds_path, transform=transform_test)
class_names_list = [d for d in os.listdir(test_ds_path) if os.path.isdir(os.path.join(test_ds_path, d))]
test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size)
text_file.write(f"Best model: Test accuracy = {calculate_accuracy(best_model, test_dataloader, device):.4f}%")

confusion_matrix = calculate_confusion_matrix(best_model, test_dataloader, device, num_classes)
fig = plt.figure(figsize = (10,7))
df_cm = pd.DataFrame(confusion_matrix, index=class_names_list, columns=class_names_list)
seaborn.heatmap(df_cm, annot=True)
plt.show()
fig.savefig(os.path.join(save_folder, "confusion matrix.png"))
text_file.close()