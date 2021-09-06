# License Plate Generator
Python version used.
```
$ python -v
$ Python 3.9.5
```
##Overview
This project generates automatically a random Peruvian License Plate. Like the following.
![](plates/AEF068.jpg)

## Usage
1. Install the dependencies listed in requirements.txt in your environment
2. Run the file ```generator_basic.py``` and specify the following parameters:
   - Image directory for saving new images
     ```-i or --img_dir``` 
   - Number of images to be generated 
     ```-n or --num```
   - To save or only show the images, is True by default
     ```-s or --save```

An example call could be like this:
```
   python generator_basic.py -i plates/ -n 1
```
