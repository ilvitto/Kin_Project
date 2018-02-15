# Kin_Project

This is a python application for reidentification of people in videos with Kinect v2 sensor.
Each video includes many people (one by one) that they pass in front of the camera for few seconds and they leave the scene. The software can recognize when is the first time the person enter or if the person is already passed before.

The application also use old data to identify and recognize subjects that are already analyzed in previous video also using different clothes or aspect.

The application shows the analyzed video and on demand open a parallel window where it shows the corrisponding match image from the database or a simple image to say that is a new subject. 

The example include only one video for dataset caused by memory problem.

## What does it use to recognize people?

It extracts some features from the video and from the Kinect results like the joints, colors, etc.. and it computes these properties to determine a global descriptor for each frame of each person. Now it use the KMeans classification to determine the number of people in the scene and compute a the "GAP Statistics" to choose the best number of cluster.

## Requirements

To use the application you have to install a python interpreter and the following packages:

* [gapkmean]: you can get it from https://pypi.python.org/pypi/gapkmean/1.0
* [scikit-learn]
* [ffmpeg]
* [numpy, openCV, pylab, shutil, matplotlib, tk]

## How to try it

To start the application just run the main file specifying the path of the videos folder and the name of the used dataset like this:

```
$ python main.py ./dataset 0078
```

You can also add the parameter "clear" to delete the dataset and start a new computation:

```
$ python main.py C:/project/dataset_folder 0078 clear
```

## Authors
* **Lorenzo Niccolai**
* **Fabio Vittorini**

# Enjoy
