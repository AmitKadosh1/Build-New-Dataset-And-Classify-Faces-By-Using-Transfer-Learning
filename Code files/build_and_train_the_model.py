import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import time

#https://www.kaggle.com/code/leifuer/intro-to-pytorch-loading-image-data
#https://discuss.pytorch.org/t/loading-dating-without-a-separate-train-test-directory-pytorch-imagefolder/130656
#https://medium.com/dejunhuang/learning-day-57-practical-5-loss-function-crossentropyloss-vs-bceloss-in-pytorch-softmax-vs-bd866c8a0d23
#https://stackoverflow.com/questions/55675345/should-i-use-softmax-as-output-when-using-cross-entropy-loss-in-pytorch
#https://pytorch.org/tutorials/beginner/saving_loading_models.html
torch.cuda.init() #It's fixed KeyError:'allocated_bytes.all.current' when use torch.cuda.memory_summary(device)

class our_model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
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
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.layers(x)

def set_parameter_requires_grad(model, feature_extract=False):
    if feature_extract: # frozen model
        for param in model.parameters():
            param.requires_grad = False
    else: # fine-tuning
        for param in model.parameters():
            param.requires_grad = True

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model = None
    weights = 'DEFAULT' if use_pretrained else None #new method from torchvision >= 0.13
    #to use other checkpoints than the default ones, check the model's available chekpoints here:
    #https://pytorch.org/vision/stable/models.html
    if model_name == "our_model":
        model = our_model(num_classes)

    elif model_name == "resnet18": #Resnet18
        model = models.resnet18(weights=weights) #new method from torchvision >= 0.13
        # model_ft = models.resnet18(pretrained=use_pretrained) #old method for toechvision < 0.13
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)  #replace the last FC layer

    elif model_name == "alexnet": #Alexnet
        model = models.alexnet(weights=weights) #new method from torchvision >= 0.13
        # model_ft = models.alexnet(pretrained=use_pretrained) #old method for toechvision < 0.13
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vgg16": #VGG16
        model = models.vgg16(weights=weights) #new method from torchvision >= 0.13
        # model_ft = models.vgg16(pretrained=use_pretrained) #old method for toechvision < 0.13
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'squeezenet1_0': #Squeezenet
        model = models.squeezenet1_0(weights=weights) #new method from torchvision >= 0.13
        # model_ft = models.squeezenet1_0(pretrained=use_pretrained) #old method for toechvision < 0.13
        set_parameter_requires_grad(model, feature_extract)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes

    elif model_name == "densenet121": #Densenet
        model = models.densenet121(weights=weights) #new method from torchvision >= 0.13
        # model_ft = models.densenet121(pretrained=use_pretrained) #old method for toechvision < 0.13
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    else:
        raise NotImplementedError
    return model

def calculate_accuracy(model, dataloader, device):
    model.eval()
    total_samples = 0
    total_correct = 0
    with torch.no_grad():
        for images,labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            _ ,prediction = torch.max(output.data, 1)
            total_samples += labels.size(0)
            total_correct += torch.sum(prediction == labels)
    accuracy = (total_correct/total_samples)*100
    return accuracy

def train_model(model, train_dataloader, val_dataloader, num_epochs, criterion, optimizer, device, save_folder):
    start_time = time.time()
    train_accuracy_list = []
    val_accuracy_list = []
    best_val_accuracy = 0  # best_val_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        model.train()
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_accuracy = calculate_accuracy(model, train_dataloader, device)
        val_accuracy = calculate_accuracy(model, val_dataloader, device)
        train_accuracy_list.append(train_accuracy.cpu().detach().numpy())
        val_accuracy_list.append(val_accuracy.cpu().detach().numpy())
        print(f"Epoch {epoch}: train_accuracy = {train_accuracy:.4f}%, val_accuracy = {val_accuracy:.4f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(save_folder, "best_model.pth"))

    end_time = time.time()
    print(f"Model training time: {end_time - start_time:.2f} secs")
    fig = plt.figure(figsize=(5, 5), tight_layout='tight')
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title('Train and validation accuracy')
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Accuracy [%]')
    axis.plot(range(1, num_epochs + 1), train_accuracy_list, label='Train accuracy')
    axis.plot(range(1, num_epochs + 1), val_accuracy_list, label='Validation accuracy')
    axis.legend()
    plt.show()
    fig.savefig(os.path.join(save_folder, "Train and validation accuracy.png"))

class AddGaussianNoise(object):
    def __init__(self, a,mean=0, std=1):
        self.mean = mean
        self.std = std
        self.a = a

    def __call__(self, tensor):
        return tensor + self.a * torch.empty(tensor.size()).normal_(self.mean,self.std)

def main():
    project_folder = "\\".join(os.path.dirname(os.path.realpath(__file__)).split("\\")[0:-1])
    models_and_results_folder = os.path.join(project_folder, "Models and Results")
    if not os.path.exists(models_and_results_folder):
        os.mkdir(models_and_results_folder)
    nums_models_in_folder = [int(d) for d in os.listdir(models_and_results_folder)]
    num_of_last_model = max(nums_models_in_folder) if (len(nums_models_in_folder) != 0) else 0
    num_model = num_of_last_model+1
    save_folder = os.path.join(models_and_results_folder, str(num_model))
    os.mkdir(save_folder)

    #Hyper parameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 300
    num_classes = len([d for d in os.listdir(os.path.join(project_folder, "Dataset Folders", "Final dataset", "train"))])
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model_name = "resnet18"
    feature_extract = False

    #transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor()])
    #transforms.Compose([transforms.Resize(224), transforms.ToTensor(), AddGaussianNoise(0.01)])
    #transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

    transform_train = transforms.Compose(
        [transforms.ToTensor()]) if model_name == "our_model" else transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(),
         transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    transform_val = transforms.Compose(
        [transforms.ToTensor()]) if model_name == "our_model" else transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

    train_ds = datasets.ImageFolder(os.path.join(project_folder, "Dataset Folders", "Final dataset", "train"), transform=transform_train)
    val_ds = datasets.ImageFolder(os.path.join(project_folder, "Dataset Folders", "Final dataset", "val"), transform=transform_val)

    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    model = initialize_model(model_name, num_classes, feature_extract)
    model = model.to(device)
    print(model)

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
    # Must write .to(device) to avoid from "RuntimeError: Input type (torch.cuda.FloatTensor) and weight
    # type (torch.FloatTensor) should be the same"

    model.load_state_dict(torch.load(os.path.join(save_folder, "best_model.pth")))
    text_file = open(os.path.join(save_folder, "Results.txt"), 'w')
    text_file.write(f"Model_name: {model_name}\nFeature extract: {feature_extract}\n")
    text_file.write(f"Training transformation: {transform_train}\n")
    text_file.write("Apply the normalization on train, validation and test\n")
    text_file.write(f"Training Parameters: Num epochs={num_epochs}, Learning rate={learning_rate}, Batch size={batch_size}\n")
    text_file.write(f"Best model: Train accuracy = {calculate_accuracy(model, train_dataloader, device):.4f}%, "
          f"Validation accuracy = {calculate_accuracy(model, val_dataloader, device):.4f}%\n")
    text_file.close()

    import test_the_model #Run test_the_model.py
if __name__ == "__main__":
    main()