# Severstal: Steel Defect Detection

Abstract: this repo includes a pipeline using Catalyst for training UNet with different encoders for the problem of steel defect detection.
Moreover, weights for trained models are provided, the result are:
- UNet with ResNet-50 - IoU 0.413
- UNet with EfficientNet-B3 - IoU 0.541
- UNet with EfficientNet-B4 - IoU 0.592

**Important:** balanced (in the meaning of defect classes) dataset includes 1000 images, where each class includes roughly 250 images.
With the whole dataset the metrics might be better.

## Plan of research
First, let's identify the main architecture. 
UNet is obviously better for this problem than Mask R-CNN.
I've conducted a research on several Kaggle kernels and papers from sources like arxiv.com.
 - Architecture: UNet
 - Encoder: EfficientNet-B3,B4; ResNet-50
 - Loss function: DiceBCELoss, TverskyLoss (alpha=0.1, beta=0.9)
 - Optimizer: Adam (learning rate for encoder 1e-3, learning rate for decoder 1e-2), as encoder is much deeper
 - learning scheduler: ReduceLROnPlateau(factor=0.15, patience=2)

## General thoughts

Important to notice that we have quite imbalanced dataset in the meaning of classes defect/no_defect.
Thus, it is importsnt to pick the appropriate loss.
I've tried DiceBCELoss and Tversky Loss (alpha=0.1 and beta=0.9).
The best results have been obtained with DiceBCELoss in this case.

Both of the encoders were pretrained on ImageNet. However, I do believe there is one more trick that can be fruitful: we can fine-tune encoders on the whole dataset (classification defect/no_defect). This way we can get some better results, but there were no images of class no_defect at all.

Also, in this situation of imbalanced classes there is point in using only images including True Positive. But, as I said above, there were no other pictures at all in the train.csv.

I need to add I've been bounded with Cuda memory capacity, so basicaly I could not try bigger encoders for batch size > 8.


## Results

| Encoder | IoU | DiceBCELoss | Mask Resolution | Epochs |
| ------ | ------ | ------ | ------ | ------ |
| ResNet-50 | 0.4132 |     | (256, 1600) |            |    |
| EfficientNet-B3  | 0.513  |      0.444        | (256, 768)| 11 |
| EfficientNet-B4  | 0.597 | 0.36 | (256, 768) |    37     |

**Link to TensorBoard for EfficientNet-B4:** [tap here](https://tensorboard.dev/experiment/rTq70zmmRJeXbklyFQs46g/#scalars)

Inferences for validation data:

 - EfficientNet-B4

 Example 1:
 ![alt text](/images/pic_1.png)

 Example 2: 
 ![alt text](/images/pic_2.png)
 
 Example 3: 
 ![alt text](/images/pic_3.png)
 
 Example 4: 
 ![alt text](/images/pic_4.png)


## Installation

Required libraries are [catalyst](https://nodejs.org/), [segmentation_models](https://github.com/qubvel/segmentation_models.pytorch) and [albumentations](https://github.com/albu/albumentations).

P.S. I've used [segmentation_models](https://github.com/qubvel/segmentation_models.pytorch) for fast prototyping.

Installation:

```sh
!pip install git+https://github.com/qubvel/segmentation_models.pytorch
!pip install -U git+https://github.com/albu/albumentations 
!pip install catalyst
```
# Usage

**The directory tree should be:**

<pre>
├── Predict_masks.py
├── Train.py
├── config.py
├── data
│   ├── results                #results
│   ├── test.csv
│   ├── test_images            #download test images here
│   ├── train.csv
│   ├── train_balanced.csv
│   └── train_images           #download train images here
├── images
│   
├── readme.md
├── utils
│   ├── losses.py
│   └── utils.py
└── weights
    ├── UnetEfficientNetB4_IoU_059.pth
    └── UnetResNet50_IoU_043.pth 
</pre>

## Evaluation

There is a Predict_masks.py script which can be used to evaluate the model and predict masks for the test dataset. The weights are stored in the ./weights directory.

It is necessary to point the directory where the test dataset is stored. Predicted masks will be stored in ./evaluation folder.

**Important:** masks for ResNet-50 are of (256, 1600)px and masks for EfficientNet-B3,B4 are of (256, 768)px. Free Colab doesn't allow to use more Cuda memory:(

Usage example:

```sh
python3 Predict_masks.py -dir /Users/user/Documents/steel_defect_detection/data/  -weights_dir /Users/user/Documents/steel_defect_detection/data/weights
```
### Arguments
```sh
-dir    : Pass the full path of a directory containing a folder "train" and "train.csv".
-num_of_images   : Number of test image from test.csv for segmentation.
-weights_dir   : Pass a weights directory.
```
Predict.py doesn't save binary masks, it saves pictures with image and predicted mask for better presentation.
## Training

The model is supposed to be trained on the dataset from the Kaggle competition. 
You can choose which encoder to use and a batch size. The default is EfficientNet-B4.
Mask size is set as (256, 768) in config.py, you can set your own.

It is necessary to point the directory where the train folder and train.csv are stored.

Usage example:
```sh
python3 Train.py -dir /Users/user/Documents/steel_defect_detection/data/ -num_of_workers 4
```
### Arguments
```sh
-dir    : Pass the full path of a directory containing a folder "test" and "test.csv".
-encoder   : Backbone to use as encoder for UNet, default='efficientnet-b3'.
-batch_size   : Batch size for training, default=8.
-num_of_workers   : Number of workers for training, default=0.
```
[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
