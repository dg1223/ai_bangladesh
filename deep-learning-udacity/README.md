# Deep Learning with PyTorch

## L3: Introduction to Neural Networks

**Maximum likelihood**: The product of the classification probabilities of every point in a dataset. This gives the likehood of the effectiveness of the entire model. Thus, we can calculate the likehood of different models and pick the one that is the most likely to give accurate predictions.

**Cross-entropy**: Taking the negative log of the maximum likelihood equation for a model gives us the cross-entropy. Since taking porducts of probabilities generate very small numbers for large datasets, it is better to take the logarithms so that we can use the sum of the individual probabilities using Information Threory concepts. The numbers are also bigger.

![Cross-entropy](https://github.com/dg1223/ai_bangladesh/blob/master/deep-learning-udacity/images/cross-entropy.png)

*From the lecturer*: If we have a number of events and their probabilities, then how likely is it that each of those events happened based on their probabilities? If it’s very likely, then we have a small cross-entropy. If it’s unlikely, then we have a large cross-entropy.

![Cross-entropy-2](https://github.com/dg1223/ai_bangladesh/blob/master/deep-learning-udacity/images/cross-entropy-2.png)

Cross-entropy =<br/>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{i=1}^{n}\sum_{j=1}^{m}{y_{ij}}ln({p_{ij}})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{i=1}^{n}\sum_{j=1}^{m}{y_{ij}}ln({p_{ij}})" title="\sum_{i=1}^{n}\sum_{j=1}^{m}{y_{ij}}ln({p_{ij}})" /></a><br/>
where:<br/>
&nbsp; &nbsp; &nbsp;n = number of events (event = a probability to predict),<br/>&nbsp; &nbsp; &nbsp;m = number of classes,<br/>&nbsp; &nbsp; &nbsp;y = [0, 1]; this is the numeric value of the label (e.g. a ‘yes, no’ label can be assigned as ‘0, 1’, making it numeric),<br/>&nbsp; &nbsp; &nbsp;p = probability of the event for the given class

> Cross-entropy is inversely proportional to the total probability of an outcome.

*Cross-entropy error*: It is also considered an error (function) because an **error is basically the negative logarithm of the probability of an event.**

### Error function

Error function =<br/>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<a href="https://www.codecogs.com/eqnedit.php?latex=-\frac{1}{m}\sum_{i=1}^{m}(1-y_{i})(ln(1-\hat{y}_{i}))&plus;y_{i}ln(\hat{y}_{i})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?-\frac{1}{m}\sum_{i=1}^{m}(1-y_{i})(ln(1-\hat{y}_{i}))&plus;y_{i}ln(\hat{y}_{i})" title="-\frac{1}{m}\sum_{i=1}^{m}(1-y_{i})(ln(1-\hat{y}_{i}))+y_{i}ln(\hat{y}_{i})" /></a><br/>
where:<br/>
&nbsp; &nbsp; &nbsp;m = number of events,<br/>&nbsp; &nbsp; &nbsp;event = a probability to predict, usually for a single data point or sample (or a row in a csv file)

The probability of a predicted output <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}" title="\hat{y}" /></a> (pronouced as y hat):<br/><br/>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}=&space;\sigma&space;(wx&plus;b)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}=&space;\sigma&space;(wx&plus;b)" title="\hat{y}= \sigma (wx+b)" /></a><br/><br/>
&nbsp; &nbsp; &nbsp;σ = sign for sigmoid function

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;![sigmoid-function](https://github.com/dg1223/ai_bangladesh/blob/master/deep-learning-udacity/images/sigmoid-function.png)

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Figure 1: The sigmoid function (σ) provides an output value between<br/>&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;0 and 1; hence used to predict probability

Therefore, the complete error function is:<br/><br/>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<a href="https://www.codecogs.com/eqnedit.php?latex=-\frac{1}{m}\sum_{i=1}^{m}(1-y_{i})(ln(1-\sigma&space;(wx_{i}&plus;b)))&plus;y_{i}ln(\sigma&space;(wx_{i}&plus;b))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?-\frac{1}{m}\sum_{i=1}^{m}(1-y_{i})(ln(1-\sigma&space;(wx_{i}&plus;b)))&plus;y_{i}ln(\sigma&space;(wx_{i}&plus;b))" title="-\frac{1}{m}\sum_{i=1}^{m}(1-y_{i})(ln(1-\sigma (wx_{i}+b)))+y_{i}ln(\sigma (wx_{i}+b))" /></a>

where:<br/>
&nbsp; &nbsp; &nbsp;w = weight<br/>
&nbsp; &nbsp; &nbsp;b = bias<br/>
&nbsp; &nbsp; &nbsp;x = training sample<br/>

We minimize this error function using the gradient descent algorithm.<br/>

## L6: Convolutional Neural Networks
* The values in an edge detection kernel (a CNN filter) should sum to 0.
* If the values do not sum up to 0, they will create the effect of brighetining or darkening the image.
* **Rule of thumb 1**: The more information you have in an image (e.g. the higher the resolution and/or frequency), the more hidden layers you might need for classification. If an image contains very little information (e.g. 28x28 MNIST images with little to no spatial information), you should use few hidden layers, probably 1 to 2 hidden layers max. Otherwise, your network might start losing important information.
