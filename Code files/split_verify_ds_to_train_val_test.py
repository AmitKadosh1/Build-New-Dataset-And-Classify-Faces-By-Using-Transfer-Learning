import os
import cv2
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

#https://www.kaggle.com/code/leifuer/intro-to-pytorch-loading-image-data

project_folder = "\\".join(os.path.dirname(os.path.realpath(__file__)).split("\\")[0:-1]) #Find main folder path (when this code is in "Code files" folder)
datasets_folder = os.path.join(project_folder, "Dataset Folders") #Build path to "Dataset Folders" folder
source_dir_path = os.path.join(project_folder, "Dataset after verify") #Building the path to the images folder after the manual verification
dest_dir_path = os.path.join(project_folder, "Final dataset") #Building the path to the final dataset folder (after splitting)

if os.path.exists(dest_dir_path): #If the destination folder exists
    shutil.rmtree(dest_dir_path) #Delete the folder
os.mkdir(dest_dir_path) #We will create the destination folder

#Open 3 folders for train, val, test in dest_dir
splitted_dir_names = ["train","val","test"]
for splitted_dir_name in splitted_dir_names:
    splitted_dir_path = os.path.join(dest_dir_path, splitted_dir_name)
    if os.path.exists(splitted_dir_path):
        shutil.rmtree(splitted_dir_path)
    os.mkdir(splitted_dir_path)

class_dir_names_list = [d for d in os.listdir(source_dir_path) if os.path.isdir(os.path.join(source_dir_path, d))]
for class_dir_name in class_dir_names_list:
    source_class_dir_path = os.path.join(source_dir_path, class_dir_name)
    #Finding all files from the source folder
    files_names_list = [f for f in os.listdir(source_class_dir_path) if os.path.isfile(os.path.join(source_class_dir_path, f))]
    image_indexes = np.arange(len(files_names_list)) #Building an array of indexes in the size of the amount of images in the source folder
    rand_gen = np.random.RandomState(0)
    rand_gen.shuffle(image_indexes) #Shuffle the list of indexes (shuffle the images before dividing)
    # Dividing the indices into 60% for Train, 20% for Validation and 20% for Test
    train_indexes, test_indexes = train_test_split(image_indexes, test_size=0.4)
    test_indexes, val_indexes = train_test_split(test_indexes, test_size=0.5)
    arrays_indexes = {"train": train_indexes, "val": val_indexes, "test": test_indexes}
    for splitted_dir_name in splitted_dir_names:
        dest_class_dir_path = os.path.join(dest_dir_path, splitted_dir_name, class_dir_name)
        if os.path.exists(dest_class_dir_path):
            shutil.rmtree(dest_class_dir_path)
        os.mkdir(dest_class_dir_path)
        for index in arrays_indexes[splitted_dir_name]: #The distribution of files according to the mixed index sets
            file_name_to_copy = files_names_list[index] #The name of the file to copy
            file_path_to_copy = os.path.join(source_dir_path, class_dir_name, file_name_to_copy) #Path to file to copy
            image_to_copy = cv2.imread(file_path_to_copy) #Reading the image to copy
            cv2.imwrite(os.path.join(dest_class_dir_path, file_name_to_copy), image_to_copy) #Saving the image in the new path in the department folders





