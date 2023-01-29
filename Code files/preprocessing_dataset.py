import os
import cv2
import shutil
import numpy as np
import imutils #Need to download this module separately

#Relevant links:
#https://realpython.com/face-recognition-with-python/
#https://stackoverflow.com/questions/30508922/error-215-empty-in-function-detectmultiscale
#https://stackoverflow.com/questions/36242860/attribute-error-while-using-opencv-for-face-recognition
#https://stackoverflow.com/questions/4195453/how-to-resize-an-image-with-opencv2-0-and-python2-6
#https://docs.opencv.org/4.5.3/d3/df2/tutorial_py_basic_ops.html#:%7E:text=making%20borders%20for%20images%20(padding)
#https://www.askpython.com/python/examples/rotate-an-image-by-an-angle-in-python
#https://datascience.stackexchange.com/questions/86402/how-to-get-pixel-location-in-after-rotating-the-image

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") #Create the haar cascade
project_folder = "\\".join(os.path.dirname(os.path.realpath(__file__)).split("\\")[0:-1]) #Find main folder path (when this code is in "Code files" folder)
datasets_folder = os.path.join(project_folder, "Dataset Folders") #Build path to "Dataset Folders" folder
source_dir_path = os.path.join(datasets_folder, "Original Dataset") #Build path to "Original Dataset" folder
dest_dir_path = os.path.join(datasets_folder, "Dataset after preprocessing") #Building the path where the images will be saved after finding the faces

if os.path.exists(dest_dir_path): #If the destination folder already exists, we will delete it and recreate it
    shutil.rmtree(dest_dir_path) #Deleting the folder
os.mkdir(dest_dir_path) #Creating the folder

wanted_height = 256 #The height of the face image to which we will convert the image using resize
wanted_width = 256 #The width of the face image to which we will convert the image using resize
class_dir_names_list = [d for d in os.listdir(source_dir_path) if os.path.isdir(os.path.join(source_dir_path, d))] #Go through all the photo folders and saving the list of classes

for class_dir_name in class_dir_names_list: #Go through all class folders
    source_class_dir_path = os.path.join(source_dir_path, class_dir_name)
    files_names_list = [f for f in os.listdir(source_class_dir_path) if os.path.isfile(os.path.join(source_class_dir_path, f))]
    files_indexes = np.arange(len(files_names_list)) #Building an array of indexes in the size of the number of images in the folder
    for file_index in files_indexes: #Go through all the images
        file_path = os.path.join(source_class_dir_path, files_names_list[file_index]) #Building a path to the image
        new_file_path = os.path.join(source_class_dir_path, "c" + str(file_index) + ".jpg") #Changing all file names to numbers plus the letter c
        #
        os.rename(file_path, new_file_path) #Changine the file name
    #We added the letter c to the numbers in case an image in the folder already has the name of some number.
    #In this situation it will not be possible to change the name of the image to the same name
    files_names_list = [f for f in os.listdir(source_class_dir_path) if
                        os.path.isfile(os.path.join(source_class_dir_path, f))]
    #At this point it is known that there is no image name that is just a number and we can convert all file names to numbers
    for file_index in files_indexes:
        file_path = os.path.join(source_class_dir_path, files_names_list[file_index])
        new_file_path = os.path.join(source_class_dir_path, str(file_index) + ".jpg")
        os.rename(file_path, new_file_path)

for class_dir_name in class_dir_names_list:
    dest_class_dir_path = os.path.join(dest_dir_path, class_dir_name)
    if os.path.exists(dest_class_dir_path):
        shutil.rmtree(dest_class_dir_path)
    os.mkdir(dest_class_dir_path) #Open a folder for the class inside the destination folder
    source_class_dir_path = os.path.join(source_dir_path, class_dir_name)
    files_names_list = [f for f in os.listdir(source_class_dir_path) if os.path.isfile(os.path.join(source_class_dir_path, f))]
    files_indexes = np.arange(len(files_names_list))
    for file_index in files_indexes: #Go through all the images in a certain class
        file_name = files_names_list[file_index] #Finding the current file name
        file_name_without_extension = "".join(file_name.split('.')[0:-1]) #Extract the file name without the extension
        file_extension = file_name.split('.')[-1] #Extract the file extension only
        source_file_path = os.path.join(source_dir_path, class_dir_name, file_name)
        image = cv2.imread(source_file_path) #Reading the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Convert the image to grayscale for the face detection algorithm
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=1, minSize=(60, 60),
                                             flags=cv2.CASCADE_SCALE_IMAGE) #Running the face detection algorithm on the grayscale image
        for num_face, (x, y, w, h) in enumerate(faces): #Go over all face frames found
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            image_height = image.shape[0] #Finding the height of the current image
            image_width = image.shape[1] #Finding the width of the current image
            face_image = image[y:y + h, x:x + w, :] #Cutting the face image from the color image according to the frame that the algorithm found
            face_image = cv2.resize(face_image, (wanted_width, wanted_height)) #Resize the face image to the size we determined above
            dest_file_name = file_name_without_extension + "-" + str(num_face) + "." + file_extension
            cv2.imwrite(os.path.join(dest_class_dir_path, dest_file_name), face_image) #Saving the face image in the appropriate folder
        #The cyclic image rotation process for the purpose of finding additional face images that were not discovered
        angle = 0
        while (angle < 360): #As long as the rotation angle is less than 360
            angle += 15 #Adding 15 degrees to the rotation angle of the current iteration
            rotate_image = imutils.rotate(image, angle=angle) #Rotate the image
            gray_rotate_image = cv2.cvtColor(rotate_image, cv2.COLOR_BGR2GRAY) #Converting the image to grayscale for entering into the face recognition algorithm
            faces = faceCascade.detectMultiScale(gray_rotate_image, scaleFactor=1.3, minNeighbors=6, minSize=(60, 60),
                                                 flags=cv2.CASCADE_SCALE_IMAGE) #Running the face detection algorithm on the grayscale image
            #Again we will save all the face images we found for the current rotation angle
            for num_face, (x, y, w, h) in enumerate(faces):
                image_height = image.shape[0]
                image_width = image.shape[1]
                face_image = rotate_image[y:y + h, x:x + w, :]
                face_image = cv2.resize(face_image,(wanted_width,wanted_height))
                dest_file_name = file_name_without_extension + "-" + str(num_face) + "." + file_extension
                cv2.imwrite(os.path.join(dest_class_dir_path, dest_file_name), face_image)
    print(f"Finish {class_dir_name}")