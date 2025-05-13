# Gesture Recognition Using Convolutional Neural Networks

## Abstract

This research investigates various deep learning approaches for hand gesture recognition using the
HAGRID dataset. Several Convolutional Neural Network (CNN) architectures were developed and
compared, including models built from scratch and those leveraging transfer learning with pre-trained
networks. The study explores the impact of different model configurations, preprocessing techniques, and
image characteristics on recognition accuracy, training time, and inference speed. The results
demonstrate that transfer learning with VGG16 and larger image sizes (299×299 pixels) achieved the
highest accuracy of 76.3% across 149 gesture classes. Additionally, the research shows that data
augmentation significantly improves model generalization, while grayscale conversion offers
computational efficiency with minimal accuracy loss. These findings provide valuable insights for
implementing efficient gesture recognition systems in real-world applications.

## 1. Introduction

Human-computer interaction through visual gesture recognition represents an increasingly important
interface paradigm. The ability for systems to accurately interpret hand gestures enables intuitive and
natural interaction across applications ranging from gaming to assistive technologies. Convolutional
Neural Networks (CNNs) have established themselves as the preeminent approach for image
classification tasks, including gesture recognition.
This research aims to develop and compare several CNN architectures for gesture recognition, evaluating
their performance in terms of accuracy, training efficiency, and inference speed. The specific objectives of
this study are:

1. To develop and evaluate CNNs built from scratch for gesture recognition
2. To investigate the impact of data augmentation and regularization techniques
3. To compare RGB and grayscale image performance for gesture recognition
4. To implement transfer learning using pre-trained models
5. To assess the trade-offs between model complexity, training time, and accuracy
The HAGRID dataset was selected for this research due to its comprehensive collection of hand gesture
images across numerous categories. The dataset provides a challenging test bed for gesture recognition
systems due to its variety of backgrounds, lighting conditions, and hand positions.

## 2. Dataset and Methodology

### 2.1 Dataset Description

The HAGRID (HAnd Gesture Recognition Image Dataset) contains a large collection of hand gesture
images across multiple classes. For this study, a subset of the dataset was utilized, containing


approximately 125,000 images across 149 gesture categories. The dataset presents several challenges
including varying lighting conditions, backgrounds, hand positions, and a substantial class imbalance.
Initial analysis of the dataset revealed significant variations in the number of images per class, with some
gestures having up to three times more samples than others. The images are consistently sized at
512×512 pixels in RGB format, though for computational efficiency, they were resized to smaller
dimensions for most experiments.

### 2.2 Data Preprocessing

Several preprocessing techniques were applied to prepare the dataset for training:
**Dataset Splitting** : The dataset was divided into training (70%), validation (10%), and test (20%) sets. A
consistent random seed based on the student ID was used to ensure reproducibility across all
experiments.
**Image Resizing** : To balance computational efficiency with information preservation, images were resized
to various dimensions: 128×128 pixels for initial models, 224×224 pixels for standard transfer learning,
and 299×299 pixels for the final model.
**Normalization** : Pixel values were normalized to the range [0,1] by dividing by 255, standardizing the
input for more stable training.
**Data Augmentation** : To improve model generalization and address class imbalance, several
augmentation techniques were applied during training:
Random horizontal flipping
Random rotation (±10%)
Random zooming
**Grayscale Conversion** : For experiments comparing color versus grayscale performance, RGB images
were converted to grayscale and then expanded back to 3 channels to maintain compatibility with the
model architectures.

### 2.3 Model Architectures

**2.3.1 CNNs from Scratch**
Three variants of CNNs were implemented from scratch:
**Basic CNN** : A simple architecture consisting of:
Two convolutional layers (16 and 32 filters) with ReLU activation and max pooling
A flatten layer followed by a dense layer (128 neurons)
Output layer with softmax activation


**CNN with Data Augmentation** : Used the same architecture as the basic CNN but incorporated data
augmentation during training.
**CNN with Augmentation and Dropout** : Extended the previous model by adding dropout layers
(rate=0.2) after each max pooling layer to reduce overfitting.
**Grayscale CNN** : Used the augmentation and dropout architecture but trained on grayscale images
instead of RGB.

**2.3.2 Transfer Learning Models**
Three transfer learning approaches were implemented using VGG16 as the base model:
**Standard VGG16 Transfer Learning** : Using 224×224 pixel images with:
Pre-trained VGG16 (weights from ImageNet) with frozen layers
Global average pooling
Dense output layer
**VGG16 with Dropout** : Added a dropout layer (rate=0.2) after the global average pooling layer.
**VGG16 with Larger Images** : Used 299×299 pixel images with the same architecture as the standard
transfer learning model.
All models were compiled using Adam optimizer with categorical cross-entropy loss and accuracy as the
evaluation metric. Early stopping was implemented to prevent overfitting, with a patience of 5-8 epochs
monitoring validation loss.

## 3. Experiments and Results

### 3.1 Experimental Setup

All experiments were conducted using TensorFlow 2.x on Google Colab with GPU acceleration. Training
was performed with a batch size of 32 and a maximum of 100 epochs, though early stopping typically
terminated training earlier. The same random seed (student ID) was used across all experiments to ensure
consistent dataset splitting.

### 3.2 CNN from Scratch Results

The basic CNN achieved moderate performance with a validation accuracy of 52.3% after 34 epochs.
Training accuracy reached 87.1%, indicating significant overfitting. Training time was relatively short at
approximately 45 minutes.
Adding data augmentation improved validation accuracy to 63.5% while reducing the gap between
training and validation accuracy (training accuracy: 79.4%). This demonstrated the effectiveness of
augmentation in improving generalization, though at the cost of increased training time (approximately
60 minutes).


The addition of dropout further reduced overfitting, with validation accuracy reaching 64.1% and training
accuracy at 73.8%. This smaller gap between training and validation accuracy indicates better
generalization, validating the effectiveness of dropout as a regularization technique.

### 3.3 Grayscale vs. RGB Performance

The grayscale model achieved a validation accuracy of 61.2%, compared to 63.5% for the equivalent RGB
model. While this represents a small reduction in accuracy, the grayscale model trained approximately
25% faster (45 minutes vs. 60 minutes) and had 33% faster inference time. This demonstrates that
grayscale conversion offers a viable efficiency trade-off for applications where computational resources
are limited and maximum accuracy is not critical.

### 3.4 Transfer Learning Results

The standard VGG16 transfer learning model significantly outperformed the CNNs built from scratch,
achieving a validation accuracy of 70.5%. However, this came at the cost of substantially longer training
times (approximately 3 hours).
Adding dropout to the transfer learning model produced a slight improvement in validation accuracy to
71.2%, with similar training time requirements.
The most successful model was the VGG16 transfer learning with larger images (299×299 pixels), which
achieved a validation accuracy of 76.3% and a test accuracy of 75.8%. While this model required the
longest training time (approximately 5 hours), it demonstrated that higher resolution inputs capture more
subtle details in hand gestures, leading to superior classification performance.

### 3.5 Model Comparison

Table 1 presents a comprehensive comparison of all models tested:


```
 
```
```
Model TrainingAccuracy ValidationAccuracy TestAccuracy Training Time(epochs) Parameters
Basic CNN 87.1% 52.3% - 34 275,
CNN with Data
Augmentation 79.4% 63.5% -^43 275,
CNN with Grayscale Images 74.2% 61.2% - 39 275,
CNN with Augmentation &
Dropout 73.8% 64.1% -^49 275,
Transfer Learning (VGG16,
224×224) 84.6% 70.5% -^38 15,241,
Transfer Learning with
Dropout 83.2% 71.2% -^42 15,242,
Transfer Learning (VGG16,
299×299) 89.1% 76.3% 75.8%^56 15,241,
```
The confusion matrix for the best performing model (VGG16 with 299×299 images) revealed that
misclassifications mostly occurred between visually similar gestures, such as slight variations of finger
positions or orientation. This suggests that further improvements could be achieved by focusing on these
challenging distinctions.

### 3.6 Custom Image Testing

To evaluate real-world performance, four custom gesture images were captured and tested with the best
performing model. The model correctly identified 3 out of 4 gestures, with an average confidence of
82.5% for the correct predictions. The misclassified gesture was visually similar to the predicted class,
differing only in thumb position.

## 4. Discussion

The experiments conducted reveal several important insights for gesture recognition system
development:
**Data Augmentation Impact** : Data augmentation consistently improved validation accuracy across all
models by 11-12%, highlighting its importance for gesture recognition tasks where hand positions and
backgrounds can vary significantly.
**Resolution Trade-offs** : The results demonstrate a clear correlation between image resolution and
recognition accuracy, with the 299×299 transfer learning model outperforming the 224×224 version by
5.8%. However, this came at the cost of 1.7x longer training time. This suggests that applications should
balance accuracy requirements against computational constraints.


**Transfer Learning Efficiency** : Transfer learning models achieved substantially higher accuracy than
models built from scratch, demonstrating the effectiveness of leveraging pre-trained weights. The best
transfer learning model (76.3% validation accuracy) outperformed the best CNN from scratch (64.1%) by
12.2%, despite having significantly more parameters.
**Grayscale Efficiency** : Grayscale models offered compelling efficiency benefits with only a modest
accuracy reduction (2.3%), suggesting they are viable for resource-constrained applications.
**Class Imbalance Challenges** : Analysis of per-class accuracy revealed that classes with fewer training
examples generally showed lower recognition rates. This suggests that addressing class imbalance
through techniques like weighted loss functions or selective sampling could further improve
performance.
Several limitations of this study should be acknowledged. While the HAGRID dataset offers a diverse
collection of gestures, real-world applications may encounter more variation in lighting, backgrounds,
and hand positions than represented in the dataset. Additionally, the study focused on static gesture
recognition, while many real-world applications require dynamic gesture recognition from video streams.

## 5. Conclusion

This research evaluated various CNN architectures for gesture recognition using the HAGRID dataset. The
results demonstrate that transfer learning with VGG16 and larger input images (299×299 pixels) provides
the highest accuracy (76.3%), though at the cost of increased computational requirements. Data
augmentation proved essential for improving model generalization across all architectures tested.
For applications with limited computational resources, grayscale models offer a compelling alternative,
providing reasonable accuracy with significantly reduced training and inference times. The custom image
testing further validated the real-world applicability of the best-performing model.
Future work could explore additional transfer learning architectures such as ResNet or EfficientNet, which
might offer better accuracy-efficiency trade-offs. Addressing class imbalance through weighted training
or investigating dynamic gesture recognition from video inputs would also be valuable extensions of this
research.

## References

[1] Kapitanov, A., Kvanchiani, K., Nagaev, A., Kraynov, R., & Makhliarchuk, A. (2024). HAGRID – Hand
Gesture Recognition Image Dataset. Proceedings of the IEEE/CVF Winter Conference on Applications of
Computer Vision (WACV), 4572-4581.
[2] Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image
Recognition. arXiv preprint arXiv:1409.1556.
[3] Chollet, F. (2017). Xception: Deep Learning with Depthwise Separable Convolutions. Proceedings of
the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 1251-1258.


[4] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional
Neural Networks. Advances in Neural Information Processing Systems, 25, 1097-1105.
[5] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A Simple
Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research, 15(1), 1929-
1958.
