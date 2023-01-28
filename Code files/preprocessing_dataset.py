import os
import cv2
import shutil
import numpy as np
import imutils #Need to download this module separately

#https://realpython.com/face-recognition-with-python/
#https://stackoverflow.com/questions/30508922/error-215-empty-in-function-detectmultiscale
#https://stackoverflow.com/questions/36242860/attribute-error-while-using-opencv-for-face-recognition
#https://stackoverflow.com/questions/4195453/how-to-resize-an-image-with-opencv2-0-and-python2-6
#https://docs.opencv.org/4.5.3/d3/df2/tutorial_py_basic_ops.html#:%7E:text=making%20borders%20for%20images%20(padding)
#https://www.askpython.com/python/examples/rotate-an-image-by-an-angle-in-python
#https://datascience.stackexchange.com/questions/86402/how-to-get-pixel-location-in-after-rotating-the-image

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") #Create the haar cascade
project_folder = "\\".join(os.path.dirname(os.path.realpath(__file__)).split("\\")[0:-1])
datasets_folder = os.path.join(project_folder, "Dataset Folders")
source_dir_path = os.path.join(datasets_folder, "Original Dataset")
dest_dir_path = os.path.join(datasets_folder, "Dataset after preprocessing")

if os.path.exists(dest_dir_path):
    shutil.rmtree(dest_dir_path)
os.mkdir(dest_dir_path)

wanted_height = 256
wanted_width = 256
class_dir_names_list = [d for d in os.listdir(source_dir_path) if os.path.isdir(os.path.join(source_dir_path, d))]

for class_dir_name in class_dir_names_list:
    source_class_dir_path = os.path.join(source_dir_path, class_dir_name)
    files_names_list = [f for f in os.listdir(source_class_dir_path) if os.path.isfile(os.path.join(source_class_dir_path, f))]
    files_indexes = np.arange(len(files_names_list))
    for file_index in files_indexes:
        file_path = os.path.join(source_class_dir_path, files_names_list[file_index])
        new_file_path = os.path.join(source_class_dir_path, "c" + str(file_index) + ".jpg")
        os.rename(file_path, new_file_path)
    files_names_list = [f for f in os.listdir(source_class_dir_path) if
                        os.path.isfile(os.path.join(source_class_dir_path, f))]
    for file_index in files_indexes:
        file_path = os.path.join(source_class_dir_path, files_names_list[file_index])
        new_file_path = os.path.join(source_class_dir_path, str(file_index) + ".jpg")
        os.rename(file_path, new_file_path)


for class_dir_name in class_dir_names_list:
    dest_class_dir_path = os.path.join(dest_dir_path, class_dir_name)
    if os.path.exists(dest_class_dir_path):
        shutil.rmtree(dest_class_dir_path)
    os.mkdir(dest_class_dir_path)
    source_class_dir_path = os.path.join(source_dir_path, class_dir_name)
    files_names_list = [f for f in os.listdir(source_class_dir_path) if os.path.isfile(os.path.join(source_class_dir_path, f))]
    files_indexes = np.arange(len(files_names_list))
    files_without_faces_indexes = []
    for file_index in files_indexes:
        file_name = files_names_list[file_index]
        file_name_without_extension = "".join(file_name.split('.')[0:-1])
        file_extension = file_name.split('.')[-1]
        source_file_path = os.path.join(source_dir_path, class_dir_name, file_name)
        image = cv2.imread(source_file_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #for find faces
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=1, minSize=(60, 60),
                                             flags=cv2.CASCADE_SCALE_IMAGE)

        faces_found = 0
        for num_face, (x, y, w, h) in enumerate(faces):
            faces_found += 1
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            image_height = image.shape[0]
            image_width = image.shape[1]
            face_image = image[y:y + h, x:x + w, :]
            face_image = cv2.resize(face_image, (wanted_width, wanted_height))
            dest_file_name = file_name_without_extension + "-" + str(num_face) + "." + file_extension
            cv2.imwrite(os.path.join(dest_class_dir_path, dest_file_name), face_image)

        angle = 0
        while (angle < 360):  # do rotation
            angle += 15
            rotate_image = imutils.rotate(image, angle=angle)
            gray_rotate_image = cv2.cvtColor(rotate_image, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray_rotate_image, scaleFactor=1.3, minNeighbors=6, minSize=(60, 60),
                                                 flags=cv2.CASCADE_SCALE_IMAGE)

            for num_face, (x, y, w, h) in enumerate(faces):
                faces_found += 1
                #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                image_height = image.shape[0]
                image_width = image.shape[1]
                face_image = rotate_image[y:y + h, x:x + w, :]
                face_image = cv2.resize(face_image,(wanted_width,wanted_height))

                dest_file_name = file_name_without_extension + "-" + str(num_face) + "." + file_extension
                cv2.imwrite(os.path.join(dest_class_dir_path, dest_file_name), face_image)

        if (faces_found == 0):
            files_without_faces_indexes.append(file_index)
    print(f"Finish {class_dir_name}")