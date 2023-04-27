# cifar10-classification

In this project, I used a part of the Resnet18 model for cifar10 classification. As you can see in the jupyter notebook file, the best accuracy was 83.54%.
Unfortunately, I could not use the tensorboard because of the attribute object error of numpy. I searched a lot and I understood that I had to update my numpy library. I did that but the error keeps arising. However, I did not remove the commands and commented them.

# Getting started
## Requirements
The file requirements.txt contains the list of required modules.

``` 
pip install -r requirements.txt 
```

# Training, Testing and Inference
For training you need to run the below command:
```
python main.py --mode train --epochs 25
```

For testing you need to run this:
```
python main.py --mode test
```

For using inference method which you can give an image to it and get the class with its probability, you have to run this:
```
python main.py --mode inference
```

Then you will be asked about your image path.
