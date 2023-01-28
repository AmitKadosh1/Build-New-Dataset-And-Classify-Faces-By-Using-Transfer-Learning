import os
import cv2
import shutil
import numpy as np
import imutils #Need to download this module separately

project_folder = "\\".join(os.path.dirname(os.path.realpath(__file__)).split("\\")[0:-1])
datasets_folder = os.path.join(project_folder, "Dataset Folders")
original_dataset_path = os.path.join(datasets_folder, "Original Dataset")
dataset_after_verify_path = os.path.join(datasets_folder, "Dataset after verify")
text_file = open(os.path.join(datasets_folder, "Unrecognized faces.txt"), 'w')

class_dir_names_list = [d for d in os.listdir(original_dataset_path) if os.path.isdir(os.path.join(original_dataset_path, d))]
for class_dir_name in class_dir_names_list:
    original_class_dir_path = os.path.join(original_dataset_path, class_dir_name)
    original_files_names_list = [f for f in os.listdir(original_class_dir_path) if
                                 os.path.isfile(os.path.join(original_class_dir_path, f))]
    after_verify_class_dir_path = os.path.join(dataset_after_verify_path, class_dir_name)
    after_verify_class_files_names_list = [f.split('-')[0] for f in os.listdir(after_verify_class_dir_path) if
                                 os.path.isfile(os.path.join(after_verify_class_dir_path, f))]
    text_file.write(f"{class_dir_name}: ")
    for f in original_files_names_list:
        if f.split('.')[0] not in after_verify_class_files_names_list:
            text_file.write(f"{f}, ")
    text_file.write("\n")
