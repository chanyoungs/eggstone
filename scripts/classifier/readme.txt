'classifier.py' is the main python file to run which has dependency of 'preprocessor.py' and 'settings.txt'
You can run 'demo.py' to see it in action with demo data on 'data_demo/'.
You can open 'demo.py' to see how classifier is used.

Basically, the module 'classifier.py' has a main class 'Predict' which takes optional input boolean argument 'setup'.
If set to true, it runs 'setup.py' in which the model, model type, parameter settings and the threshold probability is chosen through commandline.

The class initialises the prediction model which takes a few seconds to finish.

The class has one method 'predict' which takes:
Compulsory argument 'img' which is (n, n, 3) numpy array of values 0~255.
It also has two optional boolean arguments 'show_image', 'print_details' and one argument 'label' which takes in numpy array of labels for the data which can be used to check for the validity of the predictions for tests.
