import os

project_folder = "\\".join(os.path.dirname(os.path.realpath(__file__)).split("\\")[0:-1]) #Finding the main project folder
datasets_folder = os.path.join(project_folder, "Dataset Folders")
original_dataset_path = os.path.join(datasets_folder, "Original Dataset") #Path to the original images folder
dataset_after_verify_path = os.path.join(datasets_folder, "Dataset after verify") #Path to the images folder after user's manual authentication
#Opening a text file for writing in order to save the names of the files from which faces were not recognized
text_file = open(os.path.join(datasets_folder, "Unrecognized faces.txt"), 'w')

class_dir_names_list = [d for d in os.listdir(original_dataset_path) if os.path.isdir(os.path.join(original_dataset_path, d))]
for class_dir_name in class_dir_names_list: #Go through all the departments for the purpose of comparing the files
    original_class_dir_path = os.path.join(original_dataset_path, class_dir_name)
    original_files_names_list = [f for f in os.listdir(original_class_dir_path) if
                                 os.path.isfile(os.path.join(original_class_dir_path, f))]
    after_verify_class_dir_path = os.path.join(dataset_after_verify_path, class_dir_name) #Build a list of all the files in the original department folder
    after_verify_class_files_names_list = [f.split('-')[0] for f in os.listdir(after_verify_class_dir_path) if
                                 os.path.isfile(os.path.join(after_verify_class_dir_path, f))] #Building a list of all the files in the folder after the manual verification
    text_file.write(f"{class_dir_name}: ") #Writing the class name to the file
    for f in original_files_names_list:
        if f.split('.')[0] not in after_verify_class_files_names_list: #If no face image is found from a given source image
            text_file.write(f"{f}, ") #We will write the file name to the text file
    text_file.write("\n")
