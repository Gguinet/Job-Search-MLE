Links:
* [Deep Learning Interview Q&A Book](https://arxiv.org/pdf/2201.00650.pdf)

Q&A:
* How is a decision tree created? 
    * [Link1](https://www.kdnuggets.com/2020/01/decision-tree-algorithm-explained.html)
    * Gini index and Node Entropy assist the binary classification tree to make decisions. Basically, the tree algorithm determines the feasible feature that is used to distribute data into the most genuine child nodes. According to the Gini index, if we arbitrarily pick a pair of objects from a group, then they should be of identical class and the probability for this event should be 1. The following are the steps to compute the Gini index:

    1. Compute Gini for sub-nodes with the formula: The sum of the square of probability for success and failure (p^2 + q^2)
    2. Compute Gini for split by weighted Gini rate of every node of the split

    Now, Entropy is the degree of indecency that is given by the following (Where a and b are the probabilities of success and failure of the node)
    When Entropy = 0, the node is homogenous
    When Entropy is high, both groups are present at 50–50 percent in the node.
    Finally, to determine the suitability of the node as a root node, the entropy should be very low.
    
* How is a decision tree pruned?*
* Pruning is what happens in decision trees when branches that have weak predictive power are removed in order to reduce the complexity of the model and increase the predictive accuracy of a decision tree model. Pruning can happen bottom-up and top-down, with approaches such as reduced error pruning and cost complexity pruning.
Reduced error pruning is perhaps the simplest version: replace each node. If it doesn’t decrease predictive accuracy, keep it pruned. While simple, this heuristic actually comes pretty close to an approach that would optimize for maximum accuracy.

* Explain why you need VAE’s over Autoencoders. What is the loss you minimize?
    * VAE is an autoencoder whose encodings distribution is regularised during the training in order to ensure that its latent space has good properties allowing us to generate some new data.
    * so it’s used ia generative fashion.
    * KL divergence of latent layer wrt gaussian + reconstruction on last layer
    * [Link1](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
    
* What is batch normalization? How does a batch normalization layer help?
    * describe batch data, describe batch normalization
    * While the effect of batch normalization is evident, the reasons behind its effectiveness remain under discussion. It was believed that it can mitigate the problem of internal covariate shift, where parameter initialization and changes in the distribution of the inputs of each layer affect the learning rate of the network. Recently, some scholars have argued that batch normalization does not reduce internal covariate shift, but rather smooths the objective function, which in turn improves the performance. However, at initialization, batch normalization in fact induces severe gradient explosion  in deep networks, which is only alleviated by skip connections in residual networks.Others sustain that batch normalization achieves length-direction decoupling, and thereby accelerates neural networks. 
    
* Difference between LSTM’s and GRU’s.
    * The key difference between GRU and LSTM is that GRU's bag has two gates that are reset and update while LSTM has three gates that are input, output, forget. GRU is less complex than LSTM because it has less number of gates.
    * [Link1](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
    * The few differencing points are as follows:
        The GRU has two gates, LSTM has three gates
        GRU does not possess any internal memory, they don’t have an output gate that is present in LSTM
        In LSTM the input gate and target gate are coupled by an update gate and in GRU reset gate is applied directly to the previous hidden state. In LSTM the responsibility of reset gate is taken by the two gates i.e., input and target. 
        
* What does peephole keyword in TensorFlow mean for LSTM’s?
    * it’s a different version of LSTM with small gate modification
    
* What are the different types of text representations used? What is the difference between Word2Vec, Glove, FastText?
    * All are context independent embeddings [Link1](https://medium.com/analytics-vidhya/word-embeddings-in-nlp-word2vec-glove-fasttext-24d4d4286a73)
    * Word2Vec
        * *Do co-occurences + Autoencoder*
        * CBOW (input is co-occurences) vs Skip-gram (nput is word)
    * Glove
        * *Glove is a word vector representation method where training is performed on aggregated global word-word co-occurrence statistics from the corpus*.
        * *Global matrix factorization*
    * FastText is unique because it can *derive word vectors for unknown words or out of vocabulary words* — this is because by taking morphological characteristics of words into account, it can create the word vector for an unknown word. Since *morphology refers to the structure or syntax of the words*, FastText tends to perform better for such task, *word2vec *perform better for *semantic task*.
     * mix of morpho and auto-encoders
        
* What is the difference between Eigen Value Decomposition (EVD) and Singular Value Decomposition (SVD)? When does SVD behave the same as EVD?
    * SVD is EVX always possible, beyond n*n
    * SVD is when a matrix is diagonalizable and in which case it’s ODO
    
* Give me some typical Kernels ?
    * nonlinear, polynomial, Gaussian kernel, Radial basis function (RBF), sigmoid [Link1](https://data-flair.training/blogs/svm-kernel-functions/)
    
* Why there is a need for padding in CNN’s?
    * on the size of the image, to incorporate it
    
* Explain ResNet architecture.
    * One of the problems ResNets solve is the famous known *vanishing gradient* [Link1](https://medium.com/@anishsingh20/the-vanishing-gradient-problem-48ae7f501257). This is because when the network is too deep, the gradients from where the loss function is calculated easily shrink to zero after several applications of the chain rule. This result on the weights never updating its values and therefore, no learning is being performed.
    * skip connections
    
* Difference between Logistic Regression and Linear Regression.
    * max likelyhood vs classical
    
* What does maximizing the likelihood function for Logistic Regression mean? What happens if we include some prior over the weights of Logistic Regression? Why we assume prior over the weights?
    * you can deal with the prior through a regularization 
    
* When does MLE equal to MAP estimate?
    * uniform prior
    
* Why is it preferred to perform standardization over the features?
    * Normalization is good to use when you know that the distribution of your data does not follow a Gaussian distribution. This can be useful in algorithms that do not assume any distribution of the data like K-Nearest Neighbors and Neural Networks.
    * Standardization, on the other hand, can be helpful in cases where the data follows a Gaussian distribution. However, this does not have to be necessarily true. Also, unlike normalization, standardization does not have a bounding range. So, even if you have outliers in your data, they will not be affected by standardization.
    
* 5 assumptions of linear regression: 
    * Linear relationship
    * Multivariate normality
    * No or little multicollinearity [Link1](https://www.statisticssolutions.com/multicollinearity/)
    * No auto-correlation
    * Homoscedasticity [Link2](https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/homoscedasticity/)
    
* How would you improve a classification model that suffers from low precision?
    * The first thing that I would do is ask a couple of questions:
    * *Is it possible that the quality of the negative data that was collected worse than the quality of the positive data?* If so, the next step would be to understand why that is the case and how it can be resolved.
    * *Is the data imbalanced?* A model with low precision is a sign that the data is imbalanced and needs to be fixed either by over or undersampling.
    * The other thing that I would do is take advantage of algorithms that are most suited in dealing with imbalanced data, which are tree-boosted algorithms (XGBoost, CatBoost, LightGBM).
    
* Logistic vs SVM
    * [Link1](https://medium.com/axum-labs/logistic-regression-vs-support-vector-machines-svm-c335610a3d16)
    
* What is the difference between modeling the problem as P(y|x) and P(y,x)? What are the advantages of one over the other? Which algorithms are usually used and why?
    * discriminative vs generative
    * discriminative algo are better as classif usually
    
* What is the difference between MLE and MAP wrt to Linear Regression? What does linear regression assume for P(y|x)? What does Logistic regression assume for P(y|x)?
    * MLE is MAP with uniform prior, otherwise regularization
    * MLE is least square
    * Gaussian Noise
    * Log normal noise
    
* RL Overview:
    * [Link1](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
    * [Link2](https://smartlabai.medium.com/reinforcement-learning-algorithms-an-intuitive-overview-904e2dff5bbc)
    
* SVM:
    * Hinge loss for soft margin, different parametrization with hard margin, max margin, regul added as well
    * Sub-gradient descent or coordinate gradient to solve it, before primal dual
    
* Gaussian Mixture Models vs K-Means
    * [Link1](https://vitalflux.com/gaussian-mixture-models-what-are-they-when-to-use/)
    
* What is the Bayes Rule? Explain the terms posterior, likelihood, and prior.
    * posterior = prior *likelihood
    * P(hypothesis|Data) = P(hyp,Data)/P(Data) = P(Data|hyp)*P(hyp)/P(data)
    
* What is marginalization? Why is it required to marginalize over a variable?
    * Sum over all instances of this variable to get distribution of another variable
    
* Statistical test
    * t-test: 
        * Mean of a Gaussian distribution is equal to a certain value (null: H_{0}) vs different (<,>,!=), use Gaussian estimators and number of degree of freedom
        * 1 sample vs 2 sample (can map back to 1 sample case) https://web.mit.edu/~csvoss/Public/usabo/stats_handout.pdf
    * Chi-Square test for goodness of fit or independance (n-1 degree)
        * sum (O-E)^{2}/E → normalized variance estimate is chi square
        * Type 1 error: alpha, proba of rejecting true null hypo
        * Type 2 error: beta: proba of accepting false null hypo
    * confidence interval
        * https://stanford.edu/~shervine/teaching/cme-106/cheatsheet-statistics
        
* Why is it difficult to calculate posterior in VAE’s? What are the different approaches to calculate the posterior distribution?
    * Because of the normalization constant Z of the posterior used in the marginalization
    * Sampling
    
* What is batch gradient descent, minibatch gradient descent, and stochastic gradient descent? Which one would you prefer over the other?
    * batch gradient is over a batch//Minibatch is when you do the two
    * stochastic gradient descent is not a gradient descent
    
* Do you know anything about Hessian? Can they be utilized for faster training? What are its disadvantages?
    * instable potentially, condition number
    * second order methods as well (Newton’s step) → faster than first order + higher smoothness assymptions
    
* Time Series Analysis and Forecasting:
    * Trend, seasonality, variation [MIT Lecture](https://ocw.mit.edu/courses/14-384-time-series-analysis-fall-2013/resources/mit14_384f13_lec1/)
    * ARMA: autoregresive moving average !
      * autogressive is linear function of past value
      * mooving average is linear function of past noise
  * AMIRA = Intergrated autoregressive moving averagge
  * Covariance is cov(y_t,y_{t+k}), normalized by var 0
  
* What is ROC curve ?
    * [ROC Curve](https://intellipaat.com/blog/roc-curve-in-machine-learning/) is used to graphically represent the trade-off between true and false-positive rates.
    
* What is the difference between random forest and Gradient Bosted Machine ?
    * The main difference between a random forest and GBM is the use of techniques. Random forest [Link1](https://intellipaat.com/blog/what-is-random-forest-algorithm-in-python/) advances predictions using a technique called bagging. On the other hand, GBM advances predictions with the help of a technique called boosting.
    * *Bagging:* In bagging, we apply arbitrary sampling and we divide the dataset into N. After that, we build a model by employing a single training algorithm. Following that, we combine the final predictions by polling. Bagging helps to increase the efficiency of a model by decreasing the variance to eschew overfitting.
    * *Boosting:* In boosting, the algorithm tries to review and correct the inadmissible predictions at the initial iteration. After that, the algorithm’s sequence of iterations for correction continues until we get the desired prediction. Boosting assists in reducing bias and variance for strengthening the weak learners.
    
* What is p-value ?
    * P-value is used in decision-making while testing a hypothesis. The null hypothesis is rejected at the minimum significance level of the P-value. A lower P-value indicates that the null hypothesis is to be rejected.
    
* What is Naives Bayes ?
    * Despite its practical applications, especially in text mining, Naive Bayes is considered “Naive” because it makes an assumption that is virtually impossible to see in real-life data: the conditional probability is calculated as the pure product of the individual probabilities of components. This implies the absolute independence of features — a condition probably never met in real life. As a Quora commenter put it whimsically, a Naive Bayes classifier that figured out that you liked pickles and ice cream would probably naively recommend you a pickle ice cream.
    
* What are the Various Tests for Checking the Normality of a Dataset?*
    * In Machine Learning, checking the normality of a dataset is very important. Hence, certain tests are performed on a dataset to check its normality. Some of them are:
    * D’Agostino Skewness Test
    * Shapiro-Wilk Test
    * Anderson-Darling Test
    * Jarque-Bera Test
    * Kolmogorov-Smirnov Test
    
* What are the Two Main Types of Filtering in Machine Learning?
    * The two types of filtering are:
    * Collaborative filtering: Collaborative filtering refers to a recommender system where the interests of the individual user are matched with preferences of multiple users to predict new content.
    * Content-based filtering: Content-based filtering is a recommender system where the focus is only on the preferences of the individual user and not on multiple users.
    
* What’s your favorite algorithm, and can you explain it to me in less than a minute?*
    * Interviewers ask such machine learning interview questions to test your understanding of how to communicate complex and technical nuances with poise and the ability to summarize quickly and efficiently. While answering such questions, make sure you have a choice and ensure you can explain different algorithms so simply and effectively that a five-year-old could grasp the basics! I like (XGBOOST)[https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d]

* (NLP Questions)[https://www.mygreatlearning.com/blog/nlp-interview-questions/]

*A Sigmoid vs Relu as activation function ?
  * Sigmoid: not blowing up activation
  * Relu : not vanishing gradient 
  * Relu : More computationally efficient to compute than Sigmoid like functions since Relu just needs to pick  and not perform expensive exponential operations as in Sigmoids
  * Relu : In practice, networks with Relu tend to show better convergence performance than sigmoid. 
  * Sigmoid: tend to vanish gradient (cause there is a mechanism to reduce the gradient)
  * Relu : tend to blow up activation (there is no mechanism to constrain the output of the neuron, as itself is the output + Dying Relu problem - if too many activations get below zero then most of the units(neurons) in network with Relu will simply output zero, in other words, die and thereby prohibiting learning.(This can be handled, to some extent, by using Leaky-Relu instead.)

