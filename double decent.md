

# Double decent
This article will discuss a rather hot topic that has emerged in recent years: double decent.
This article discusses a rather hot topic that has emerged in recent years: double détente.

With the widespread use of neural networks, many have realized that with NNs, it seems better to have too many hidden units than too few. Similarly, more hidden layers are better than fewer. Running SGD until zero training error often yields good out-of-sample error. In fact, for images with high signal-to-noise ratios, such networks will run indefinitely until the training error reaches zero. Such models have a very large number of parameters. Increasing the number of units or layers and training again until zero error sometimes results in **even better** out-of-sample error.
All of this suggests that neural networks seem less prone to overfitting. you can put any number of parameters into the model. Deep learning breaks the **bias-variance tradeoff**, a core idea in traditional statistical learning used to explain the generalization ability of models.
The following article argues that, in the modern context, the bias-variance tradeoff is **Wrong** and does **not apply to NNs**.
[Reconciling modern machine learning practice and the bias-variance trade-off](https://arxiv.org/abs/1812.11118)


> Written with [StackEdit](https://stackedit.io/).
