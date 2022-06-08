---
title: "Inception scores"
date: 2022-06-08 17:00:00
categories: Deep learning
---

# Inception Score
Papers
1. Improved techniques for training GANs (https://arxiv.org/pdf/1606.03498.pdf)
2. A Note on the Inception Score (https://arxiv.org/pdf/1801.01973.pdf)


## Evaluating (Black-Box) Generative Models
In contrast to supervised learning, Generative Adversarial Networks lack an objective function that measures "What is realistic?", which makes it difficult to compare performance of different models. 


Why is it so hard?
- The real data distribution p(x) is unknown
- The explicit generative distribution q(x) is unknown. (ex. GANs uses random noise vectors for the latent variable)

## Some previous metrics for the evaluation of generative models
- To approximate density function over generated samples and then calculate the likelihood of held-out samples
- To apply a pre-trained neural network to generated images and calculate statistics of its output or at a particular hidden layer.(Inception Score approach)
- Use a crowd-sourcing platform (Amazon Mechanical Turk) to evaluate a large number of GAN generated images. 

## What is Inception Score?
The **Inception Score** is a metric for automatically evalutating the quality of image by generative adversarial networks. ([Salimans et al.2016](https://arxiv.org/pdf/1606.03498.pdf))

The Inception Score uses an Inception v3 Network pre-trained on ImageNet and calculates a statistic of the network's outputs when applied to generated images. The probability of the image belonging to each class is predicted and then, theses predictions are summarized into the Inception Score.

### Criterions
- Image Quality : The generated image should be sharp rather than blurry
- p(y|x) should be low entropy : the inception network should be highly confident there is a single object in the image
- Image Diversity : p(y) should be high entropy. (the generative algorithm should output a high diversity of images)

informally,
- Every realistic image should be recognizable, which means that the score distribution for it must be, ideally, dominated by one class.(The entropy of image-wise class distributions, p(y|x) should have low entropy)
- Class distribution over the whole sample should be as close to uniform as possible, in other words, a good generator is a diverse generator.(The entropy of the overall distribution,which is p(y), should be high)


How to Calculate?
KL divergence = p(y|x) * (log(p(y|x)) – log(p(y)))
```python
import numpy as np

def inception_score(model, images , n_split = 10 , eps = 1E-16):
    scores = []
    n_part = int(images.shape[0] / n_split)
    
    y_hat = model.predict(images)
    
    for i in range(n_split):
        ix_start , ix_end = i * n_part , ( i + 1 ) * n_part
        p_yx = y_hat[ix_start : ix_end]
        p_y = np.expand_Dims(p_yx.mean(axis = 0 ) , 0 ) 
        D_kl = p_yx * (log (p_yx + eps) - log(p_y + eps))
        D_kl_sum = D_kl.sum(axis = 1)
        scores.append( np.exp( np.mean(D_kl_sum) ) )
    scores = np.array(scores)
    return np.mean(scores) , np.std(scores)
```


## Issues With the Inception Score
1. Suboptimalities of the Inception Score itself
2. Problems with the popular usage of the Inception Score

It is extremely important when reporting the Inception Score of an algorithm to include some alternative scroe demonstrating that the model is not overfitting to training data, validating that the high score achieved is not simply replaying the training data.

# References
- https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/
