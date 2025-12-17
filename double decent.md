# Double decent

This article will discuss a rather hot topic that has emerged in recent years: double decent.
The main content comes from  [Stanford Statistical Learning course](https://learning.edx.org/course/course-v1:StanfordOnline+SOHS-YSTATSLEARNINGP+1T2024/home?audit_mode=) materials.

With the widespread use of neural networks, many have realized that with NNs, it seems better to have too many hidden units than too few. Similarly, more hidden layers are better than fewer. Running SGD until zero training error often yields good out-of-sample error. In fact, for images with high signal-to-noise ratios, such networks will run indefinitely until the training error reaches zero. Such models have a very large number of parameters. Increasing the number of units or layers and training again until zero error sometimes results in **even better** out-of-sample error.

All of this suggests that neural networks seem less prone to overfitting. you can put any number of parameters into the model. Deep learning breaks the **bias-variance tradeoff**, a core idea in traditional statistical learning used to explain the generalization ability of models.

The following article argues that, in the modern context, the bias-variance tradeoff is **Wrong** and does **not apply to NNs**.
[Reconciling modern machine learning practice and the bias-variance trade-off](https://arxiv.org/abs/1812.11118)

## Simulation

To aid understanding, We created a simulation using Google AI Studio:

The feature x is uniformly distributed between -5 and 5. The error follows a Gaussian distribution with a standard deviation of 0.3. We use a small training set with 20 samples and a **very large test set** (10k) to find the specific causes of the test errors.

```math
y = \sin(x) + \varepsilon,\quad x \sim U[-5,5],\ \varepsilon \sim \mathcal{N}(0,\,0.3^2)
```

We fit the data using a **natural spline**, a method for fitting flexible functions. It uses d degrees of freedom. This means **a linear regression fit on d basis functions**. So our prediction will be represented as a linear combination of these basis functions with d parameters.

```math
\hat{y}_i = \beta_1 N_1(x_i) + \beta_2 N_2(x_i) + \cdots + \beta_d N_d(x_i)
```

**Splines** are a class of piecewise polynomial functions. They are spliced ​​at several "knots". The function values ​​and derivatives (usually up to second order) must be continuous at the knots. The model we use is **linear with respect to the parameters but nonlinear with respect to x**.

What we do is change d. We can enrich the space by adding d.

When d=20, the number of model parameters is the same as the number of training samples, and the spline basis functions are **linearly independent**, making the design matrix full rank. Therefore, the linear regression equation has a **unique solution**, which can **accurately match** the observation value at each training sample point, so that all training residuals are equal to zero.

When d>20, we can still obtain 0 residual solutions, but they are not unique. We can obtain infinitely many solutions with zero residuals. We choose the solution with the **minimum norm**.
i.e. the zero-residual solution with smallest
$\sum_{j=1}^d \hat{\beta}_j^2$
When $d > n$, the linear least squares solution is not unique.  Among all zero-training-error solutions, we select the one with the minimum $\ell_2$ norm, known as the **minimum-norm least squares solution**.

<img src="1.png" width="500">

The above graph shows the results of the simulation. It displays a double-decreasing curve. The horizontal axis represents 'd' because we added a basis function. The vertical axis represents the error. Test error is shown in blue, and training error in orange. 

We can see the **training error** between d < 20 observations. As d increases, it continuously decreases. When d = 20, it equals 0 and remains 0 thereafter.

Interestingly, the **test error** initially decreases, then increases as overfitting of the training data begins. This part illustrates **the common trade-off** between bias and variance. Initially, errors are high because of bias. Later, they decrease. Then, due to variance, the error spikes dramatically, breaking through the ceiling.

The most interesting part happens. The error begins to decreasing again. This is the **double-decent** phenomenon.Then it seems to start rising again.

So when d > 20, the sum of squared coefficients $\sum_{j=1}^d \hat{\beta}_j^2$ is:

Despite more coefficients, the sum of squares decreases. Because we have more opportunities to fit the data, we can find configurations with smaller $\hat{\beta}$ values ​​for the sum of squares. The principle is that when we have more parameters, most of the beta becomes smaller. We will **reduce those wiggy solutions**. In other words, we separated the expressive power required by the data into more parameters. In this example, this means **separated out** into more basis functions.

The next graph shows the multiple results. The first graph, with 8 degrees of freedom, represents a good result, yielding a decent approximation.

The second graph, with 20 parameters, requires passing through every observed data point. We can see that it must** stretch **itself considerably. Because the 20-parameter model must strive to fully adapt to the data, errors elsewhere become quite large.

Near the interpolation threshold (
d≈n=20), the model is forced to interpolate all training points, which can lead to a highly oscillatory solution and poor generalization (large test error), often due to ill-conditioning and sensitivity to noise.

The third graph shows the result with 42 degrees of freedom, after double-decent. The function performs much better. These changes make the process **smoother**. Although there are slight fluctuations, hopefully not as severe as in the second graph, because the $\hat{\beta}$ value is smaller. The situation is similar in the fourth graph.

![alt text](<2.png>)

## Some facts：

In a wide linear model $(p \gg n)$ fit by least squares, SGD with a small step size leads to a **minimum-norm** zero-residual solution.

Stochastic gradient **flow** — i.e., the entire path of SGD solutions — is somewhat similar to the ridge path.

By analogy, deep and wide neural networks fit by SGD down to zero training error often give good solutions that generalize well.

In particular cases with **high signal-to-noise ratio** — e.g., image recognition — models are less prone to overfitting; the zero-error solution is mostly signal!
