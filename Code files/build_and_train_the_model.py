import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#Need: torchvision >= 0.13
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import time

#Relevant links:
#https://www.kaggle.com/code/leifuer/intro-to-pytorch-loading-image-data
#https://discuss.pytorch.org/t/loading-dating-without-a-separate-train-test-directory-pytorch-imagefolder/130656
#https://medium.com/dejunhuang/learning-day-57-practical-5-loss-function-crossentropyloss-vs-bceloss-in-pytorch-softmax-vs-bd866c8a0d23
#https://stackoverflow.com/questions/55675345/should-i-use-softmax-as-output-when-using-cross-entropy-loss-in-pytorch
#https://pytorch.org/tutorials/beginner/saving_loading_models.html

torch.cuda.init() #It's fixed KeyError:'allocated_bytes.all.current' when use torch.cuda.memory_summary(device)

class reference_model(nn.Module): #The class definition for a reference model
    def __init__(self, num_classes):
        super().__init__() #Building the network architecture
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1),      #padding=1 with kernel=3X3 give 'same' convolution
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=4),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=4),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),
            nn.Flatten(), #Flattening the information before moving to the FC layers
            nn.Linear(2048, 1024),
            #The value is 2048 because a 256x256 image enters and goes through two pollings of 4 and another one of 2, which means we get 8x8.
            #The origin of the last layer is 32 filters.
            #That's why we get 8X8X32 which after flattening you get 2048.
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
    def forward(self, x): #Implementation of the function forward which returns the origin of the network for the entry x
        return self.layers(x)

#A function that defines whether the network parameters will be trained or not
def set_parameter_requires_grad(model, feature_extract=False):
    if feature_extract: # feature extraction method
        for param in model.parameters():
            param.requires_grad = False #The parameters will not train
    else: # fine-tuning method
        for param in model.parameters():
            param.requires_grad = True #The parameters will train

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model = None #Initialize the model variable that this function will return
    weights = 'DEFAULT' if use_pretrained else None
    #to use other checkpoints than the default ones, check the model's available chekpoints here:
    #https://pytorch.org/vision/stable/models.html
    if model_name == "reference_model":
        model = reference_model(num_classes)
    #For all models:
    #torchvision >= 0.13: model = models.<network_name>(weights=weights)
    #torchvision < 0.13: model = models.<network_name>(pretrained=use_pretrained)
    elif model_name == "resnet18": #Resnet18
        model = models.resnet18(weights=weights)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes) #Replace the last FC layer
    elif model_name == "alexnet": #Alexnet
        model = models.alexnet(weights=weights)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes) #Replace the last FC layer
    elif model_name == "vgg16": #VGG16
        model = models.vgg16(weights=weights)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes) #Replace the last FC layer
    elif model_name == 'squeezenet1_0': #Squeezenet
        model = models.squeezenet1_0(weights=weights)
        set_parameter_requires_grad(model, feature_extract)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
    elif model_name == "densenet121": #Densenet
        model = models.densenet121(weights=weights)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes) #Replace the last FC layer
    else: #If the network name sent to the function is different from the options listed above
        raise NotImplementedError #throw an error
    return model

def calculate_accuracy(model, dataloader, device):
    model.eval() #Moving the model to evaluation mode (affects the normalization and dropout layers)
    total_samples = 0 #The amount of samples we have gone through so far
    total_correct = 0 #The number of times the model has hunted so far
    with torch.no_grad(): #During the accuracy calculation do not change the gradients
        for images,labels in dataloader: #Go over all the images and labels in the batch
            images = images.to(device) #Transferring the images to the device
            labels = labels.to(device) #Transferring the labels to the device
            output = model(images) #Calculation the otput of the model
            _ ,prediction = torch.max(output.data, 1) #Getting the maximum index from the output vector
            total_samples += labels.size(0) #Updating the amount of samples
            total_correct += torch.sum(prediction == labels) #Updating the number of times the model was right
    accuracy = (total_correct/total_samples)*100 #Calculation of model accuracy in percentages
    return accuracy

def train_model(model, train_dataloader, val_dataloader, num_epochs, criterion, optimizer, device, save_folder):
    start_time = time.time() #Saving the start time of the training for the purpose of calculating the training time
    train_accuracy_list = [] #A list for saving the accuracy on the train during training
    val_accuracy_list = [] #A list for saving the accuracy on the validation during training
    best_val_accuracy = 0 #Setting the best accuracy to 0 so that any other value will be higher than it
    for epoch in range(1, num_epochs + 1): #Performing the loop as many times as the number of epochs
        model.train() #Change the model to training mode
        for images, labels in train_dataloader:
            images = images.to(device) #Transferring the images to the device
            labels = labels.to(device) #Transferring the labels to the device
            output = model(images) #Calculation the otput of the model
            loss = criterion(output, labels) #Calculate Loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_accuracy = calculate_accuracy(model, train_dataloader, device)
        val_accuracy = calculate_accuracy(model, val_dataloader, device)
        train_accuracy_list.append(train_accuracy.cpu().detach().numpy())
        val_accuracy_list.append(val_accuracy.cpu().detach().numpy())
        print(f"Epoch {epoch}: train_accuracy = {train_accuracy:.4f}%, val_accuracy = {val_accuracy:.4f}%")
        if val_accuracy > best_val_accuracy: #If it is the current state of the model is better than the previous state
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(save_folder, "best_model.pth")) #Saving the best model to file
    end_time = time.time() #Saving the end time of the training for the purpose of calculating the training time
    print(f"Model training time: {end_time - start_time:.2f} secs")
    fig = plt.figure(figsize=(5, 5), tight_layout='tight')
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title('Train and validation accuracy')
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Accuracy [%]')
    axis.plot(range(1, num_epochs + 1), train_accuracy_list, label='Train accuracy')
    axis.plot(range(1, num_epochs + 1), val_accuracy_list, label='Validation accuracy')
    axis.legend() #Add a graph legend
    plt.show() #View the graph
    fig.savefig(os.path.join(save_folder, "Train and validation accuracy.png")) #Save the graph to a file

class AddGaussianNoise(object): #Creating a class to add Gaussian noise augmentation
    def __init__(self, a,mean=0, std=1): #The default is standard normal distribution
        self.mean = mean
        self.std = std
        self.a = a
    def __call__(self, tensor):
        return tensor + self.a * torch.empty(tensor.size()).normal_(self.mean,self.std)

def main():
    project_folder = "\\".join(os.path.dirname(os.path.realpath(__file__)).split("\\")[0:-1])
    models_and_results_folder = os.path.join(project_folder, "Models and Results")
    if not os.path.exists(models_and_results_folder): #If the folder does not exist
        os.mkdir(models_and_results_folder) #Create the folder
    #The folder will contain folders with names of numbers according to the model number
    nums_models_in_folder = [int(d) for d in os.listdir(models_and_results_folder)]
    #Finding the number of the last model saved in the folder
    num_of_last_model = max(nums_models_in_folder) if (len(nums_models_in_folder) != 0) else 0
    num_model = num_of_last_model+1 #Adding 1 to the last model to get the number of the new model
    save_folder = os.path.join(models_and_results_folder, str(num_model))
    os.mkdir(save_folder) #Creating a folder to save the model and important data for it

    #Hyper parameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 300
    num_classes = len([d for d in os.listdir(os.path.join(project_folder, "Dataset Folders", "Final dataset", "train"))])
    criterion = torch.nn.CrossEntropyLoss() #Suitable for multi-class classification problems
    #If there is a GPU run on it and if not then on the CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device) #Printing the device
    model_name = "resnet18" #The name of the model
    feature_extract = False #Should you use feature extraction or fine tuning?

    #Augmentations:
    #transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor()])
    #transforms.Compose([transforms.Resize(224), transforms.ToTensor(), AddGaussianNoise(0.01)])
    #transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

    #Defining transformations and augmentations on the Train
    transform_train = transforms.Compose(
        [transforms.ToTensor()]) if model_name == "reference_model" else transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor()])
    # Defining transformations and augmentations on the Validation
    transform_val = transforms.Compose(
        [transforms.ToTensor()]) if model_name == "reference_model" else transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor()])

    train_ds = datasets.ImageFolder(os.path.join(project_folder, "Dataset Folders", "Final dataset", "train"), transform=transform_train)
    val_ds = datasets.ImageFolder(os.path.join(project_folder, "Dataset Folders", "Final dataset", "val"), transform=transform_val)

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    model = initialize_model(model_name, num_classes, feature_extract)
    model = model.to(device)
    print(model) #Printing the model

    num_trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("num trainable weights: ", num_trainable_params)
    #calculate the model size on disk
    size_model = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits
    print(f"model size: {size_model} bit | {size_model/8e6:.2f} MB")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_model(model, train_dataloader, val_dataloader, num_epochs, criterion, optimizer, device, save_folder)
    model.load_state_dict(torch.load(os.path.join(save_folder, "best_model.pth"))) #Loading the best model
    #Opening a text file for writing in order to save the details of the model and its results
    text_file = open(os.path.join(save_folder, "Results.txt"), 'w')
    text_file.write(f"Model_name: {model_name}\nFeature extract: {feature_extract}\n")
    text_file.write(f"Training transformation: {transform_train}\n")
    text_file.write("Apply the normalization on train, validation and test\n")
    text_file.write(f"Training Parameters: Num epochs={num_epochs}, Learning rate={learning_rate}, Batch size={batch_size}\n")
    text_file.write(f"Best model: Train accuracy = {calculate_accuracy(model, train_dataloader, device):.4f}%, "
          f"Validation accuracy = {calculate_accuracy(model, val_dataloader, device):.4f}%\n")
    text_file.close() #Closing the text file

    import test_the_model #Run test_the_model.py and updates the text file with the results

if __name__ == "__main__": #Run the main only if the code is run from this file
    main()