# Image_Classification_with_CNN

### Clarification

- This coursework is running and testing on Google Colab. 

- Because the test set is extracted randomly from the dataset, the accuracy on the test set cannot precisely represent the performance of model. But it is still a reasonable approach for evaluating the performance of model.

- The evaluation on the test set is using the model at final epoch in task 1-3 using the model with lowest loss on validation set in task 4.

### **Task 1: Experiments**

#### **Subtask 1-1: How does the number of layers affect the training process and test performance? Try between 2 and 5 layers.**

Firstly, to clarify the architecture of 4 CNNs for the experiments, I assume the layer 4 and 5 are following the pattern of given layer 1-3 —— the output channels of each layer are added by 8 from its input channels. The parameters of each layer are listed in table 1. Each convolutional layer is followed by the same ReLU, max-pooling and dropout (see table 2). After the last convolutional layer, there are 2 fully connected layers (see table 3). Other relevant parameters are shown in table 4.

In short, the control variable is only the number of layers and its parameters, everything else, even the increasing pattern of the layer’s parameters, keep the same as possible.



Table 1: Parameters of each convolutional layer.

| Layer 1 | Input channels = 3, output channels = 16,  kernel size = 3  |
| ------- | ----------------------------------------------------------- |
| Layer 2 | Input channels = 16, output channels = 24,  kernel size = 4 |
| Layer 3 | Input channels = 24, output channels = 32,  kernel size = 4 |
| Layer 4 | Input channels = 32, output channels = 40,  kernel size = 4 |
| Layer 5 | Input channels = 40, output channels = 48, kernel  size = 4 |

Table 2: Functional layers followed by each layer, and relevant parameters.

| ReLU        |                 |
| ----------- | --------------- |
| Max-pooling | Kernel size = 2 |
| Dropout     | Rate = 0.3      |

Table 3: Fully connected layer, and relevant parameters.

| Fully connected layer 1 | Input = (flattened output of last  convolutional layer), output = 512 |
| ----------------------- | ------------------------------------------------------------ |
| Fully connected layer 2 | Input = 512, output = 10                                     |

Table 4: Other relevant parameters and methods.

| Other relevant parameters and methods.                       |
| ------------------------------------------------------------ |
| Learning  rate = 0.001, Batch size = 16, momentum=0.9  Optimizer = SGD, Loss function = CrossEntropyLoss |

 

In the subtask 1-1, table 5 compares the accuracy of CNNs with a different number of maximal layers, visualizing their confusion matrixes and Loss curves. For each CNNs, the training will stop when the model converges (pre-train the model to find out the most suitable training epochs —— the point when loss curve start increasing, which means overfitting and losing generalization on unseen data). For example, 2-layer CNN will converge at 4th epoch and achieve 49.89% accuracy on the test set, and 5-layer CNN can achieve 63.17% of accuracy on the test set, and this 5-layer CNN will converge at 40th epoch.



Table 5. Range of accuracy on the test set, after 3 times execution for each architectures (different number of maximal layers).

| Maximal  layer | The highest  accuracy (%) | The lowest  accuracy (%) | Number of  Epochs when converge |
| -------------- | ------------------------- | ------------------------ | ------------------------------- |
| Layer 2        | 49.89                     | 45.06                    | 4                               |
| Layer 3        | 53.22                     | 53.11                    | 10                              |
| Layer 4        | 57.17                     | 55.94                    | 15                              |
| Layer 5        | 63.17                     | 59.00                    | 40                              |

Convolution layer is used to abstract feature and simplify the complexity of the network. A different number of filters and layers can have different performance. The results shown in table 5 indicate that with the increment of layers mentioned above, the accuracy of the model on the test set also increased. It is reasonable in this case because more layer will output more feature map to represent the original image and for classifying

In figure 1, we can have a clearer understanding of how the models with different layer perform when predicting the class on the test set. On confusion matrix (a) (b) (c) (d), the percentages of a misclassified image are decreasing when the CNN go deeper (seeing the light cells excluding diagonal getting dark from (a) to (b)). The loss curves (e) (f) (g) (h) in which the lines tend to be flat are used to detect converge.

The best performance is given by 5-layer CNN, achieving 63.17% accuracy on the test set at 40th epochs. The parameters are listed in Table 1-5. This 5-layer CNN will be used in task 2 & 3, called CNN-5L.

![截屏2020-04-15 下午5.43.01](https://tva1.sinaimg.cn/large/007S8ZIlgy1gduxi6uxvsj31320ioe5h.jpg)

Figure 1. Confusion matrix and loss curves of CNNs at 1st execution —— (a) (e) 2-layer, (b) (f) 3-layer, (c) (g) 4-layer, (d) (h) 5-layer

#### **Subtask 2: Choose one more architectural element to test**

In subtask2, the report will test with 3 different elements. The first experiment is to test the performance of different batch size in 3-layer CNN. The second experiment is to increase the number of fully connected layers in 5-layer CNN. The third experiment is to increase the parameters for each convolutional layer.

1. The first experiment is based on 3-layer convolutional neural, keeping other parameters same as mentioned above, and train the model with different batch size (see table 6). The model will be pre-trained to find out the number of epoch to converge before the evaluation of accuracy on the test set.

   Intuitively, a larger batch size means the number of iterations required to run an epoch is reduced and therefore speed up the processing. If the batch size is within an ideal range, a larger the batch size will take more points into account, and therefore can make the gradient direction more accurate with less fluctuation. However, based on the result, to achieve similar accuracy, it needs more epochs to run.



Table 6 Comparison of accuracy and converge epochs for different batch size

| Batch size | Converge epochs | Accuracy |
| ---------- | --------------- | -------- |
| 8          | 7               | 49.44    |
| 16         | 10              | 53.22    |
| 32         | 13              | 51.28    |

 

Figure 2: Confusion matrixes of 8 batches CNN (a), 16 batches CNN (b) and 32 batches CNN (c).

2. The result of 5-layer architecture seems good. In this experiment, based on the previous 5-layer network, a new CNN with one more fully connected layer with the same 512 input and output channels is tested (see table 7.).  The accuracy of this model on the test set is 59.22% for the first time and 59.59% for the second time. It seems that the performance does not have much difference comparing with CNN-5L (the original 5-layer CNN).

![截屏2020-04-15 下午5.43.29](https://tva1.sinaimg.cn/large/007S8ZIlgy1gduxixih8lj311m0d6qoo.jpg)

Table 7. Parameters of 3 fully connected layers

| Fully connected layer 1 | Input = (flattened output of last  convolutional layer), output = 512 |
| ----------------------- | ------------------------------------------------------------ |
| Fully connected layer 2 | Input = 512, output = 512                                    |
| Fully connected layer 3 | Input = 512, output = 10                                     |

![截屏2020-04-15 下午6.00.33](https://tva1.sinaimg.cn/large/007S8ZIlgy1gduy0gqk30j30um0ec49z.jpg)

Figure 3. Confusion matrix (a) and loss curve (b) in experiment 2 (with 59.22% ACC on the test set).

3. The third experiment is to increase the parameters for each convolutional layer —— multiply by 4 (see table 8). Larger channels values normally mean that the model can extract more features from an image. After pre-trained the CNN, the loss curve shows the model will converge at around 23rd epoch. Running 23rd epoch again, the model got 61.22% accuracy on the test set. It seems slightly better than the model in experiment 2 but still stays in the potential accuracy range of CNN-5L.



Table 8. Parameters of each layer in the 5-layer CNN

| Layer 1 | Input channels = 3, output channels = 64,  kernel size = 3   |
| ------- | ------------------------------------------------------------ |
| Layer 2 | Input channels = 64, output channels = 96,  kernel size = 4  |
| Layer 3 | Input channels = 96, output channels = 128,  kernel size = 4 |
| Layer 4 | Input channels = 128, output channels = 160,  kernel size = 4 |
| Layer 5 | Input channels = 160, output channels = 196,  kernel size = 4 |

![截屏2020-04-15 下午6.01.09](https://tva1.sinaimg.cn/large/007S8ZIlgy1gduy110xrwj30uw0e0n8i.jpg)

Figure 4: Confusion matrix (a) and loss curve (b) in experiment 3.

### **Task 2: Filter visualization** 

The filters before training are initialized randomly (see figure 5-a). During the training process, the filters will be updated by gradient descent and backpropagation. In figure 5, there are significant differences between the filters before training (5-a) and halfway training (5-b), which means the model change a lot during the process. However, comparing with the filters in halfway training and after training, there only slight differences, which means that the filters of the first layer of this specific model are close to the optimal values (but possibility local optimum instead of global optimum). This phenomenon can also be observed on loss and accuracy curve (see figure 6). It is clear that the differences between 0 and 20th epochs are significant but much smaller between 20th and 40th epochs (e.g. compare the gaps between 3 lines in figure 6-a).

![截屏2020-04-15 下午6.01.39](https://tva1.sinaimg.cn/large/007S8ZIlgy1gduy1ia24zj311s0dw7jy.jpg)

Figure 5: Filters before training (a), halfway training (b) and after training (c), outputting from Conv2d without ReLU.

![截屏2020-04-15 下午6.02.35](https://tva1.sinaimg.cn/large/007S8ZIlgy1gduy2hyckcj311y0euakn.jpg)

Figure 6: Loss (a) and accuracy (b) curve in CNN-5L.

### **Task 3: Feature map visualization**

The number of feature maps equals to that of output channels, each filter will extract a type of feature map. With the network goes deeper, feature maps will become smaller and blurrier, but also means being more representative than the previous layer. For example, figure 7 (a) is showing the image of a banana. It is obvious that the feature maps at 6th row are smaller and more abstractive than the 2nd row. Loot at the graffiti on the banana peel in the original image. The information of these graffiti which is unrelated for identifying and classifying is gradually removed from the feature maps as the network goes deeper.

![截屏2020-04-15 下午6.03.21](https://tva1.sinaimg.cn/large/007S8ZIlgy1gduy3hpsi1j30u00u6b29.jpg)

![截屏2020-04-15 下午6.03.40](https://tva1.sinaimg.cn/large/007S8ZIlgy1gduy3qjgaoj313e0eejuh.jpg)

Figure 7: Original and normalized image (1), feature maps after each CONV layer without ReLU (2-6), and feature map before fully connected layer (7), banana (a) and mug (b)

### **Task 4: Improving network performance**

In this task, inspired by AlexNet, this improved model (call CNN-task4) can achieve 69.11% accuracy on the test set. The relevant structure and parameters are listed below (see table 9). Noticed that the accuracy on the test set is evaluated by the model achieving the lowest loss on validation set during training (save the model at that epoch and reload it in evaluation).



Table 9: Structure and parameters of CNN-task4

![截屏2020-04-15 下午6.04.27](https://tva1.sinaimg.cn/large/007S8ZIlgy1gduy4h8ikej31240pgaej.jpg)

One of the adjustments is using another optimizer Adam, which support adaptive learning rate, updating the learning rate during learning. The reason behind is that adaptive learning rate will slow down the stride of gradient descent during the training. Giving a small stride can help reduce the possibility of missing and fluctuating around the (local or global) optimal solution. From figure 8, we can notice that CNN-task4 with 0.0001 initial learning rate can give a better loss curve than the others.

![截屏2020-04-15 下午6.04.48](https://tva1.sinaimg.cn/large/007S8ZIlgy1gduy4s62maj310c09swh3.jpg)

Figure 8: Loss curve of original 5-layer CNN with 0.01 LR (a)，CNN-task4 with 0.001 initial LR (b)，CNN-task4 with 0.0001 initial LR (c)

Another adjustment is to move the dropout layer from convolution layers to fully connected layers. One reason is that the parameters in the convolution layer are quite few but highly relative. The pixels within a certain area of an image in convolutional layers are sharing the same information, which means that the discarded information may still be retained by nearby pixels. Using dropout layer in convolution layer may only help control the noise of the input images but not improving the generalization on unseen data (see figure 9 (d)). On the other hand, if the feature maps between 2 layers are highly abstracted when passing forward such as convolution operation with large kernel size or stride, or max-pooling operation, using dropout will, on the contrary, cause the feature maps to lose important information for classifying. And therefore, the performance of the model will decline, especially causing a significant decrease of accuracy on the test set and fluctuation of loss curve on the validation set (see figure 9 (c)).

Table 10 is comparing the accuracy on the test set of 4 different dropout layer placing strategies. The 2nd and 4th row are showing that adding dropout after convolution layer 3 & 4, which have small kernel and stride, having similar performance. However, once the dropout layers are placed after max-pooling layer (see 3rd row). The accuracy of the model on the test set will decline from around 65% to 50%.



Table 10, Position of dropout layer (based on the CNN-task4 in table 9) and the highest accuracy on the test set after 30 epochs training

| Dropout(0.5)  Before FC1&2                  | 69.11% | Figure 9 (a) |
| ------------------------------------------- | ------ | ------------ |
| No dropout                                  | 66.17% | Figure 9 (b) |
| Dropout(0.3)  after max-pooling layer 1&2&3 | 50.22% | Figure 9 (c) |
| Dropout(0.3)  after CONV 3&4                | 64.94% | Figure 9 (d) |

![截屏2020-04-15 下午6.05.16](https://tva1.sinaimg.cn/large/007S8ZIlgy1gduy5ahtb9j311608o76i.jpg)

Figure 9: Loss curve of CNN-task4, with dropout before FC1&2 (original) (a), without dropout (b)，with 0.3 dropout after max-pooling (c)，with 0.3 dropout after CONV 3&4 (d).
