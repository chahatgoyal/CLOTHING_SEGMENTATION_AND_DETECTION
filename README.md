# CLOTHINGSEGMENTATIONANDDETECTION
This model is basically used to detect clothes from human body and recognize them through fashion MNIST


# Package Requirements
1.Python
  
2.OpenCV 3.1.0
  
3.Keras with tensorflow backend
  
4.Pandas
  
5.NumPy

 The code can be used for any industry on any images and the core algortithm is 'grab-cut algorithm" with the blend of Deep-Learning Convolutional Neural Networks. The Repo is designed in a preview way and its limited for fashion Images with auto-segmenting Top-wear clothes(Example: Tshirt, shirts) and Full-body clothes(salwar,gowns, shirt-pants-shoes) After segmentation the clothes are classified according to their colour and types trained on FASHION MNIST
 



Demo (AI-Segmentaion)
Demo Annotation shouldn't be replaced, adding new will not enable the code adaptation to new classes of images.(*The demo phase classes : Fashion full-body, Top-wear*)
```
1.*clone* the Repo to your local pc ensuring that all the package requirements satisfied.<enter>
  
2.Run the code from the terminal **python fashion.py image1.jpg /Users/demo/save** <enter>
  
3.argument1 -- *image_name -- image1.jpg*, argument2 -- *save_directory -- /Users/demo/*
```
4.Visualize the results as output of OPENCV

