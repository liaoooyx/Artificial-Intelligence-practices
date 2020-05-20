# Image_Caption_Generation

[参考文章](https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8)

### Encoder-Decoder模型

图片字幕生成的细节：

- 模型使用的词典不在乎单词的顺序问题

- 这是一个监督学习问题

  - 通过输入的Xi，预测输出的Yi

    ![截屏2020-04-15 下午2.33.57](https://tva1.sinaimg.cn/large/007S8ZIlgy1gdus9eyca0j311q0h8n0h.jpg)

  - 也即是给定图片向量，每次预测基于当前的部分字幕，生成下一个单词

    - 输入：图片向量和当前的部分字幕（以词典中的索引表示）
      - 将字幕转化为输入，涉及到词嵌入技术（the word embedding techniques），比如 GLOVE词嵌入模型，将每个单词都匹配为长度200（自定义）的向量
      - 也可以单纯的只使用词典本身作为向量（在词典中存在为1，不存在为0）
    - 预测输出：下一个单词

---

### 相关问题

[参考文章](https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8)

### Encoder-Decoder模型

图片字幕生成的细节：

- 模型使用的词典不在乎单词的顺序问题

- 这是一个监督学习问题

  - 通过输入的Xi，预测输出的Yi

    ![截屏2020-04-15 下午2.33.57](https://tva1.sinaimg.cn/large/007S8ZIlgy1gdus9eyca0j311q0h8n0h.jpg)

  - 也即是给定图片向量，每次预测基于当前的部分字幕，生成下一个单词

    - 输入：图片向量和当前的部分字幕（以词典中的索引表示）
      - 将字幕转化为输入，涉及到词嵌入技术（the word embedding techniques），比如 GLOVE词嵌入模型，将每个单词都匹配为长度200（自定义）的向量
      - 也可以单纯的只使用词典本身作为向量（在词典中存在为1，不存在为0）
    - 预测输出：下一个单词

---

### 相关问题

#### **Q1. What the advantages/disadvantages might be of using lemmatized vs regular tokens.**

Tokenization is the process of splitting any string into words. It is an essential approach for data preparation relevant to field of language processing. In this process, there are lots of methods can be applied thus lead to different type of tokens such as lemmatized tokens and regular tokens. Lemmatization usually refers to transforming words from inflected, singular forms etc. to the base form, known as the lemma, which is an actual word in a dictionary. For example, *am*, *is*, *are* will be converted to *be*, and *runs*, *ran*, *running*, will be of *run*. During Lemmatization process, the transformation of a specific word relies on the current context, such as identifying whether the word *saw* is a noun or a verb.

The regular tokens are generated from simple process such as tokenization, removing punctuation, converting words into lowercase while lemmatized tokens have one more process which is Lemmatization. The mainly different is whether the token word is remaining same or converted to a base form.

There are many advantages of using lemmatized tokens. One of the most widely known advantages is it can give a better result by removing the inflectional endings only and therefore can properly represents a group of related words with same token. For example, in this Image Caption Generation task, where the data is not big enough, lemmatized tokens can represent the highly discrete words more aggregated. The inflected words with same meaning will be grouped and hence will not affect the weight differently in the network. From another perspective, it is equivalent to increasing the amount of data. As a result, the generated caption might be more accurate. However, it comes with the costs. Lemmatization uses a corpus to transform the words and takes the context of original text into account, which means more computing and storage consumption. On the other words, it will slow down the processing speed and use more memory.

On the other hand, the advantages and disadvantages of regular tokens are opposite to lemmatized tokens. As its simplicity, it is suitable for the tasks with constraints on time and memory. What’s more, English is not an inflection-rich language (comparing with Spanish and Arabic) thus not that much influence if using regular tokens. But the results may inevitably worse than using lemmatized tokens.

In conclusion, using lemmatized tokens is usually a better option, but still coms with its drawback. In some circumstances, the regular tokens might be more suitable. As a result, people should depend on the use case to choose a proper approach. 

#### **Q2. Present the sample images and generated caption for each epoch of training for both the RNN and LSTM version of the decoder, including the BLEU scores.**

In this task, the report chooses the same two images for observing the generated caption for RNN and LSTM. Table 1 is listing the parameters which remain the same in the sampling process, including the image, the reference captions used for computing BLEU score, and the weights of BLEU for each gram. Table 2-6 are comparing the generated caption and the corresponding BLEU cumulative score between LSTM and RNN at each epoch. Noticed that the tokens <start> and <end> has been removed from generated caption.

![截屏2020-04-17 下午5.07.53](https://tva1.sinaimg.cn/large/007S8ZIlgy1gdx7qbb7ghj30rs0k6gt9.jpg)

![截屏2020-04-17 下午5.08.08](https://tva1.sinaimg.cn/large/007S8ZIlgy1gdx7qm6t9xj30s408sacl.jpg)

![截屏2020-04-17 下午5.09.09](https://tva1.sinaimg.cn/large/007S8ZIlgy1gdx7rl8hmhj30p808e0xv.jpg)

![截屏2020-04-17 下午5.09.27](https://tva1.sinaimg.cn/large/007S8ZIlgy1gdx7rxx5xuj30rq0qkn56.jpg)

#### **Q3. Compare training using an RNN vs LSTM for the decoder network.**

In this coursework, we are the using encoder-decoder model for image caption generation. For both RNN decoder or LSTM decoder, they are all using the same encoder with CNN, which is responsible for extracting and compressing the contents of images into small feature vectors. And the variable that might affect the generated captions is controlled between RNN and LSTM in the decoder. The comparison will start by comparing the structure and then move to the results of the CW code, including the overall loss and BLEU score, the difference for training with long or short captions and quality of generated captions.

From the structural level, the recurrent neural network (RNN) is a feedforward neural network that has internal memory and is recurrent in nature. The traditional convolutional neural network (CNN) in which the data stream is flowing from layer to layer has no connections between neurons (or nodes) inside the layer, causing current neuron lacks memory for the previous neurons. While RNN, on the contrary, has connections between neurons inside the hidden layers. The inputs of a current neuron include the outputs of the last layer and the previous neuron in the current layer. As a result, RNN is suitable for sequence-related tasks such as speech recognition, natural language processing, etc. 

Long short-term memory network (LSTM) is a variant of RNN, belonging to feedback neural network. It is aiming to deal with the problems of gradient vanish that traditional RNN encountered during training, which means losing information in a long-distance propagation. LSTM has a similar structure as RNN in general but more complex in details. Comparing with RNN, a common LSTM has an additional parameter to keep the memory of previous data over arbitrary time intervals, called cell state, which is the core of LSTM. To regulate the information over the cell, there are three gates named forget gate, input gate and output gate. The forget gate decides which information will be removed from the last cell state. The input gate is deciding what information should be added to the current cell state. Concatenate these two steps with the last cell state, it can update the current cell state. Then, the output gate determines which information will be outputted based on above, and finally get the output. This structure helps LSTM to address the vanishing gradient problem, and accordingly, have better performance than traditional RNN in terms of longer time series tasks

To evaluate and compare models, the first criteria is observing the loss value on both the straining step and test step. Figure 1 illustrates the trends of loss for LSTM and RNN, including the loss values for every batch and the average loss values at each epoch. But the difference between them is hard to find. From table 7, which is showing the average loss in number, it can still be noticed that the loss of LSTM is slightly bigger than that of RNN at the first epoch and becomes slightly smaller at the final epoch.

| ![地图的截图  描述已自动生成](https://tva1.sinaimg.cn/large/007S8ZIlgy1gduye45gbrj30h00cw0tx.jpg) | ![手机屏幕截图  描述已自动生成](https://tva1.sinaimg.cn/large/007S8ZIlgy1gduye5z19zj30hb0crmyc.jpg) |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                         (1)    LSTM                          |                          (2)    RNN                          |

 Figure 1: Loss per batch and  average loss per epoch during training for LSTM (1)  and RNN (2).  



Table 7: Average loss for LSTM and RNN on training set at each epoch.

![截屏2020-04-15 下午6.20.37](https://tva1.sinaimg.cn/large/007S8ZIlgy1gduylad30jj310y0900ts.jpg) 

After finish training the model, we can evaluate the model using the test set. Here we compare the loss of model on the test set between LSTM and RNN, as well as the differences of BLEU score. In Table 8, the loss of model with LSTM decoder is also lower than that with RNN decoder. Figure 2 is illustrating the cumulative BLEU scores from 1-gram to 4-gram, which can describe the overall performance of generated captions in the perspective of words. The areas under the line are representing the percentage of captions with different scores, where blue lines stand for LSTM and orange lines stand for RNN. From the distribution of the area under the lines, we might find that the area of RNN is prone to the left than the area of LSTM, indicating a lower score. The vertical dash lines in the figures that point out the average score over the whole test set (figure 2, where the actual values are shown in table 9) prove this again. On the other hand, we have to notice that LSTM are more likely to get scores very close to zero.



Table 8: Loss for LSTM and RNN on test set

|          | LSTM  | RNN   |
| -------- | ----- | ----- |
| Test set | 2.603 | 2.725 |

![图片包含 游戏机, 设备  描述已自动生成](https://tva1.sinaimg.cn/large/007S8ZIlgy1gduye539qzj30z80b341c.jpg)

Figure 2: All of Cumulative BLEU scores and the average of scores on test set for LSTM and RNN, score from 0 to 1 separated into 30 bins.



 Table 9: Average cumulative BLEU score for LSTM and RNN on test set.

|        | LSTM  | RNN   |
| ------ | ----- | ----- |
| 1-gram | 0.550 | 0.494 |
| 2-gram | 0.461 | 0.416 |
| 3-gram | 0.452 | 0.417 |
| 4-gram | 0.467 | 0.442 |



Table 10 is comparing the results between images with the longest reference captions and shortest captions on average in the test set. For the image with the longest reference captions, although the 1-gram BLEU score for LSTM is slightly lower than that for RNN, the other three type of cumulative BLEU scores, from 2-gram to 4-gram, are all higher. And for the image with the shortest one, all of the BLEU scores when using LSTM to train the decoder are higher than using RNN. However, if we compare the generated captions in a human perspective, it is obvious that all of them are misdescribing the corresponding images at the same point thus do not show much difference for people.  Table 11 is showing three randomly chosen samples, giving clues that the performance of LSTM and RNN are unstable and not always giving similar scores.

We compare the loss of LSTM decoder and RNN decoder, during the training epoch on the training set and after finish training on the test set, as well as the cumulative BLEU scores on the generated captions, on average and on both the longest and the shortest reference captions. In conclusion, the difference between Loss and BLEU score in overall may indicate that LSTM may have a better performance than RNN. But with human perception, it is hard to judge which is better. And the LSTM does not show its advantages in dealing with long captions. However, this insight may not obvious and convincing enough in this case as the training set, length of the captions and training epochs seem quite constrained.



Table 10: Comparing the images with long (left) or short (right) reference captions on average in test set.

![截屏2020-04-15 下午6.21.38](https://tva1.sinaimg.cn/large/007S8ZIlgy1gduyncvc1mj30u00vu454.jpg) 

![截屏2020-04-17 下午5.10.09](https://tva1.sinaimg.cn/large/007S8ZIlgy1gdx7svcdz8j30ru13u4by.jpg)

#### **Q4. Among the text annotations files downloaded with the Flickr8k dataset are two files we did not use: ExpertAnnotations.txt and CrowdFlowerAnnotations.txt. Read the readme.txt to understand their contents, then consider and discuss how these might be incorporated into evaluating your models.**

The file ExpertAnnotations.txt contains a set of image-captions pairs with a score that rated by experts from 1 (the caption does not describe the image at all) to 4 (the caption describes the image correctly). The file CrowdFlowerAnnotations.txt contains a collection of image-captions pairs with the judgement, by asking human whether the caption describes the image or not. Both of them giving additional non-correct captions for images compared with the file Flickr8k.token.txt, while the Crowd Flower Annotations use a binary judgement and the Expert Annotations provides a finer-grained score.

One of the limitations within the encoder-decoder model is that the only connection between the encoder and decoder is a fixed-length feature vector which represents the images with an uncertain number of objects. The feature vector may not represent all of the information of the input image so that the accuracy of the decoder will hence be affected.

On the other hand, the domain of image caption generation is related to the supervised learning using an algorithm to find out the optimal solution for generating appropriate caption of a specific image based on a collection of caption-image pairs. It is similar to image classification but far more complex as there are more objects. Hodosh et al. (2013) [1] shows that using multiple captions for an image gives better results than using a single caption. Extending from this, we suppose that if the model has additional data about not only what is correct but also what could be wrong, it might be possible to improve the model and achieve better results.

For example, when converting the words to the vector at the decoder step, we can introduce finer-grain weights with non-correct caption instead of binary values based on the two additional files. Or use it to improve the performance of the word embedding functions.

 

Reference:

[1] M. Hodosh, P. Young, and J. Hockenmaier. Framing image description as a ranking task: Data, models and evaluation metrics. J. Artif. Int. Res., 47(1):853–899, May 2013.


