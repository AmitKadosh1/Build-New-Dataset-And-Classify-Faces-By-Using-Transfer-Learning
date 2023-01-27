# Build NewnDataset And Classify Faces By Using Transfer Learning
This is a class project as part of EE046211 - Deep Learning course @ Technion.  

<p align="center">
Batel Shuminov</a>  •  
Amit Kadosh</a>



## Introduction
Face classification is an important capability that can have a wide variety of uses. for example:
1. The system can be used to help blind people know who is at the front door of the house.
2. The system can also be integrated into the smart home field - automatic opening of the house door or simply reading the name of the person standing at the entrance to the house.
3. Creating personal albums - you can locate the photos in which a certain person appears and thus build a personal album for him easily.
This area was not tested due to the lack of a suitable dataset.
In addition, face classification is an individual problem of each person because each person knows different people and wants to classify them.
Therefore, we found it appropriate to prove the feasibility of implementing a system based on Deep Learning for the purpose of face classification for a group of people on which the model will be trusted.
The main difficulty expected in this project is that Deep Learning works well when there are many examples and in this project the number of examples is very few.
In order to overcome this difficulty, we will use transfer learning which was trained on another dataset for a different purpose.



## Dataset
### Original Dataset
The name Original Dataset represents the original images loaded into the array before any processing.
The dataset for our project should contain images of some persons with different characteristics (for example, different situation, different lighting, different camera and more). There is no Dataset like this around the internet so we had to build it.
In order for this project to be able after further hardware development to be used by anyone, we need to require a relatively low minimum threshold of images for each person. Therefore, we chose to use 50 images for each person.
We built the dataset for this project from images from our personal smartphones and social networks. We collect data of 15 classes (different people) with each having only 50 images.
We will save the images in a folder called "Original Dataset" which contains 15 folders for each of the people we want to teach our model.

<p align="center">

![Default](./examples/examples1.jpg)
<p align="center">
Figure 1: Example of images from "Amit" class in "Original Dataset".

### Prepressing Dataset
#### Face Recognition
Now we will move to the second stage - the face recognition stage in the image.
We used a built-in algorithm in Python called Cascade Classifier from a GitHub project which is based on classic Machine Learning. This algorithm has been widely used in many web projects.
OpenCV uses machine learning algorithms to search for faces within a picture. For something like a face, might have 6,000 or more classifiers, all of which must match for a face to be detected (within error limits). the algorithm starts at the top left of a picture and moves down across small blocks of data, looking at each block, and check if is this a face. Since there are 6,000 or more tests per block, might have millions of calculations to do. To get around this, OpenCV uses cascades. The OpenCV cascade breaks the problem of detecting faces into multiple stages. For each block, it does a very rough and quick test. If that passes, it does a slightly more detailed test, and so on. The algorithm may have 30 to 50 of these stages or cascades, and it will only detect a face if all stages pass. The advantage is that the majority of the picture will return a negative during the first few stages, which means the algorithm won’t waste time testing all 6,000 features on it. Instead of taking hours, face detection can now be done in real time.
This algorithm works with grayscale images, so in order to run the algorithm, we first converted the images to grayscale.
After receiving the face location from the algorithm, we cut the face from the original color image. Finally, we resized it to 256x256 so that all the face photos would be the same size.

<p align="center">

![Default](./examples/examples2.jpg)
<p align="center">
Figure 2: Face Recognition. (a) The original image, (b) The image in gray scale, (c) Face recognition performed by the Cascade Classifier algorithm, (d) Cropping the face from original image and resize to 256x256.








