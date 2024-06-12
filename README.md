## PatchLine/Patch predictor
---

## File Download
The pretrained patch predictor weights can be downloaded from Baidu Cloud.\
The weights are derived from multi-vision tasks for classification(VGG16), object detection(SSD-VGG16), and semantic segmentation(U-Net-VGG16) with the VGG16 backbone.\
Link: https://pan.baidu.com/s/1ix36PZDEk3NuacFhkBLDKQ?pwd=hmzs 
Extraction code: hmzs


The complete test dataset used for evaluation can also be downloaded from Baidu Cloud.\
The test data are derived from Core-50 dataset with the VGG16 backbone.\
The VGG16 contains three convolutional groups, the images are divided into three categories(patch-1/patch-2/patch-3), each representing the need for 1, 2, or 3 patches.\
Link: https://pan.baidu.com/s/1iDPCbowYhDfAe-f14soSww?pwd=apds 
Extraction code: apds



## Training 
1. The images in the datasets folder are divided into two parts: train contains the training images, and test contains the test images.
2. Before training, you need to prepare the dataset by creating different folders in the train or test directory. Each folder should be named after the corresponding class, and the images in the folder should belong to that class. 
3. After preparing the dataset, run txt_annotation.py in the root directory to generate the cls_train.txt needed for training. Before running, modify the classes to match the classes you need.
4. Then, modify the cls_classes.txt file in the model_data folder to correspond to the classes you need.
5. After adjusting the network and weights in train.py, you can start training!

## Prediction
### a、Using Pretrained Weights
1. After downloading the weights, place them in the "model_data" directory. The test images are in "test_demo_patch_select". Run predict.py and input the image path to predict the required number of patches.
### b、Using Your Own Trained Weights
1. Train according to the training steps.
2. In the classification.py file, modify model_path, classes_path, backbone, and alpha to correspond to the trained files; model_path corresponds to the weights file in the logs folder, classes_path is the class file corresponding to model_path, backbone is the backbone feature extraction network used, and alpha is the alpha value when using mobilenet.  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   Modify model_path and classes_path to use your trained model for prediction!
    #   model_path points to the weight file in the logs folder, classes_path points to the txt file in model_data
    #   If there is a shape mismatch, also pay attention to the modifications of model_path and classes_path parameters during training
    #--------------------------------------------------------------------------#
    "model_path"    : 'model_data/best_epoch_weights.pth',
    "classes_path"  : 'model_data/cls_classes_patch.txt',
    #--------------------------------------------------------------------#
    #   Input image size
    #--------------------------------------------------------------------#
    "input_shape"   : [350, 350],
    #--------------------------------------------------------------------#
    #   Model types used:
    #--------------------------------------------------------------------#
    "backbone"      : 'vgg16',
    #-------------------------------#
    #   Whether to use Cuda
    #   If no GPU is available, set to False
    #-------------------------------#
    "cuda"          : True
}
```
3. Run predict.py, input image path

## Evaluation 
1. The images in the datasets folder are divided into two parts: train contains the training images, and test contains the test images. During evaluation, we use the images in the test folder.
2. Before evaluation, you need to prepare the dataset by creating different folders in the train or test directory. Each folder should be named after the corresponding class, and the images in the folder should belong to that class. 
3. After preparing the dataset, run txt_annotation.py in the root directory to generate the cls_test.txt needed for evaluation. Before running, modify the classes to match the classes you need.  
4. Then, in the classification.py file, modify model_path, classes_path, backbone, and alpha to correspond to the trained files; model_path corresponds to the weights file in the logs folder, classes_path is the class file corresponding to model_path, backbone is the backbone feature extraction network used, and alpha is the alpha value when using mobilenet.  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   Modify model_path and classes_path to use your trained model for prediction!
    #   model_path points to the weight file in the logs folder, classes_path points to the txt file in model_data
    #   If there is a shape mismatch, also pay attention to the modifications of model_path and classes_path parameters during training
    #--------------------------------------------------------------------------#
    "model_path"    : 'model_data/best_epoch_weight.pth',
    "classes_path"  : 'model_data/cls_classes_patch.txt',
    #--------------------------------------------------------------------#
    #   Input image size
    #--------------------------------------------------------------------#
    "input_shape"   : [350, 350],
    #--------------------------------------------------------------------#
    #   Model types used:
    #--------------------------------------------------------------------#
    "backbone"      : 'vgg16',
    #-------------------------------#
    #   Whether to use Cuda
    #   If no GPU is available, set to False
    #-------------------------------#
    "cuda"          : True
}

```
5. Run eval.py to evaluate the patch predictor accuracy.

## Demo

1. Download weights of patch predictor.\
The pretrained patch predictor weights can be downloaded from Baidu Cloud.\    
Link: https://pan.baidu.com/s/1ix36PZDEk3NuacFhkBLDKQ?pwd=hmzs 
Extraction code: hmzs

2. Run eval.py with the default parameters to eval the demo dataset (test_demo_patch_select).

## Reference
https://github.com/bubbliiiing/classification-pytorch  

