## Notes

---

收集更多的训练集有时并不能带来很大作用.

#### Evaluating Hypothesis:

---

1种方法: **分成7:3 training set 和 test set.**

通过70%的training set学习 $\theta$, 就是之前讲的如何前向传播和反向传播.

再通过30%的test set得到$J_{test}(\theta)$(就是**Misclassfication error**).

##### 计算Misclassification error:

$$
\operatorname{err}\left(h_{\Theta}(x), y\right)=\begin{array}{ll}
{1} & {\text { if } h_{\Theta}(x) \geq 0.5 \text { and } y=0 \text { or } h_{\Theta}(x)<0.5 \text { and } y=1} \\
{0} & {\text { otherwise }}
\end{array}
$$

预测错了就算1, 最后除m.
$$
\text { Test Error }=\frac{1}{m_{\text {test}}} \sum_{i=1}^{m_{\text {test}}} \operatorname{err}\left(h_{\Theta}\left(x_{\text {test}}^{(i)}\right), y_{\text {test}}^{(i)}\right)
$$



---

第二种方法: **6:2:2, Training set : Cross validation set : test set**.

用validation set来select model.

先在training set训练出$\theta$. 再在validation set上测试这些训练的$\theta$, 选一个最小的.

**generalization error of the model**通过test set计算出.

上面一种方法是不好的, 因为那样虽然可以得到较低的generalization error of the model, 就是直接用test set去做最后generalization error of the model, 因为选出的是最小的

![1580128235174](E:\Coursera_AndrewNg_ML_Exercises\Coursera_AndrewNg_ML_Exercises\assets\1580128235174.png)

![1580128480887](E:\Coursera_AndrewNg_ML_Exercises\Coursera_AndrewNg_ML_Exercises\assets\1580128480887.png)



#### Bias vs. Variance:

---

CV: cross validation.

**High Bias => underfit:**

$J_{train}$ will be high.

$J_{CV}$约等于$J_{train}$.

**High Variance => overfit:**

$J_{train}$ will be low.

$J_{CV} >> J_{train}$.

![1580130045760](E:\Coursera_AndrewNg_ML_Exercises\Coursera_AndrewNg_ML_Exercises\assets\1580130045760.png)

![1580130080572](E:\Coursera_AndrewNg_ML_Exercises\Coursera_AndrewNg_ML_Exercises\assets\1580130080572.png)



#### Regularization and Bias/Variance

---

$J_{train}$不带正则惩罚项?

要理解$J_{CV}(\theta)$是二者不可兼得的, 在提高对训练集数据的拟合程度的时候 这个J **会先下降后上升**.

![1580130765720](E:\Coursera_AndrewNg_ML_Exercises\Coursera_AndrewNg_ML_Exercises\assets\1580130765720.png)



#### Deciding What to Do Next Revisited:

Our decision process can be broken down as follows:

**过拟合: 减少feature, 增加训练样例, 增加$\lambda$.**

**欠拟合: 增加feature, 增加多项式高次, 减少$\lambda$**. (增加训练样例也可).

- **Getting more training examples:** Fixes high variance

- **Trying smaller sets of features:** Fixes high variance

- **Adding features:** Fixes high bias

- **Adding polynomial features:** Fixes high bias

- **Decreasing λ:** Fixes high bias

- **Increasing λ:** Fixes high variance.

#### Diagnosing Neural Networks:

- A neural network with fewer parameters is **prone to underfitting**. It is also **computationally cheaper**.
- A large neural network with more parameters is **prone to overfitting**. It is also **computationally expensive**. In this case you can use regularization (increase λ) to address the overfitting.





#### Support Vector Machine

---

![1580266656199](E:\Coursera_AndrewNg_ML_Exercises\Coursera_AndrewNg_ML_Exercises\assets\1580266656199.png)

先看cost function和LR的相似之处.

优化(minimum)一个cost function, 前面乘一个正的常数是没事的, 所以就有了**上面的题**.

![1580266800573](E:\Coursera_AndrewNg_ML_Exercises\Coursera_AndrewNg_ML_Exercises\assets\1580266800573.png)
$$
C = \frac{1}{\lambda}
$$












#### 好题

---

![1580100046072](E:\Coursera_AndrewNg_ML_Exercises\Coursera_AndrewNg_ML_Exercises\assets\1580100046072.png)

![1580101634021](E:\Coursera_AndrewNg_ML_Exercises\Coursera_AndrewNg_ML_Exercises\assets\1580101634021.png)

![1580189438820](E:\Coursera_AndrewNg_ML_Exercises\Coursera_AndrewNg_ML_Exercises\assets\1580189438820.png)

上面是因为已经 high variance => overfitting 了, 变更复杂的神经网络没用.



![1580222236702](E:\Coursera_AndrewNg_ML_Exercises\Coursera_AndrewNg_ML_Exercises\assets\1580222236702.png)

![1580223232899](E:\Coursera_AndrewNg_ML_Exercises\Coursera_AndrewNg_ML_Exercises\assets\1580223232899.png)

![1580223864397](E:\Coursera_AndrewNg_ML_Exercises\Coursera_AndrewNg_ML_Exercises\assets\1580223864397.png)

![1580223461417](E:\Coursera_AndrewNg_ML_Exercises\Coursera_AndrewNg_ML_Exercises\assets\1580223461417.png)

![1580264323381](E:\Coursera_AndrewNg_ML_Exercises\Coursera_AndrewNg_ML_Exercises\assets\1580264323381.png)

| True/False | Answer                                                       | Explanation                                                  |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| False      | We train a learning algorithm with a small number of parameters (that is thus unlikely to overfit). |                                                              |
| True       | We train a learning algorithm with a large number of parameters (that is able tolearn/represent fairly complex functions). | You should use a "low bias" algorithm with many parameters, as it will be able to make use of the large dataset provided. If the model has too few parameters, it will underfit the large training set. |
| True       | The features x contain sufficient information to predict y accurately. (For example, one way to verify this is if a human expert on the domain can confidently predict y when given only x). | It is important that the features contain sufficient information, as otherwise no amount of data can solve a learning problem in which the features do not contain enough information to make an accurate prediction. |
| False      | We train a model that does not use regularization.           | Even with a very large dataset, some regularization is still likely to help the algorithm's performance, so you should use cross-validation to select the appropriate regularization parameter. |
| False      | The classes are not too skewed.                              | The problem of skewed classes is unrelated to training with large datasets. |
| True       | Our learning algorithm is able to represent fairly complex functions (for example, if we train a neural network or other model with a large number of parameters). | You should use a complex, "low bias" algorithm, as it will be able to make use of the large dataset provided. If the model is too simple, it will underfit the large training set. |
| False      | When we are willing to include high order polynomial features of x | As we saw with neural networks, **polynomial features can still be insufficient to capture the complexity of the data**, especially if the features are very high-dimensional. Instead, you should use a complex model with many parameters to fit to the large training set. |
| True       | A human expert on the application domain can confidently predict y when given only the features x (or more generally we have some way to be confident that x contains sufficient information to predict y accuratly) | This is a nice project commencement briefing.                |

![1580264511113](E:\Coursera_AndrewNg_ML_Exercises\Coursera_AndrewNg_ML_Exercises\assets\1580264511113.png)

| True/False | Answer                                                       | Explanation                                                  |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| True       | The classifier is likely to now have lower recall.           | Increasing the threshold means more y = 0 predictions. This increase will decrease the number of true positives and increase the number of false negatives, so recall will decrease. |
| False      | The classifier is likely to have unchanged precision and recall, but lower accuracy. | By making more y = 0 predictions, we decrease true and false positives and increase true and false negatives. Thus, precision and recall will certainly change. We cannot say whether accuracy will increase or decrease. |
| False      | The classifier is likely to have unchanged precision and recall, but thus the same F1 score. | By making more y = 0 predictions, we decrease true and false positives and increase true and false negatives. Thus, precision and recall will certainly change. We cannot say whether the F1 score will increase or decrease. |
| False      | The classifier is likely to now have lower precision.        | Increasing the threshold **means more y = 0 predictions.** This will decrease both true and false positives, so precision will increase, not decrease. |



| True/False | Answer                                                       | Explanation                                                  |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| True       | Using a **very large** training set makes it unlikely for model to overfit the training data | A sufficiently large training set will not be overfit, as the model cannot overfit some of the examples without doing poorly on the others. |
| False      | It is a good idea to spend a lot of time collecting a **large** amount of data before building your first version of a learning algorithm. | You cannot know whether a huge dataset will be important until you have built a first version and find that the algorithm has high variance. |
| False      | After training a logistic regression classifier, you **must** use 0.5 as your threshold for predicting whether an example is positive or negative. | You can and should adjust the threshold in logistic regression using cross validation data. |
| False      | If your model is underfitting the training set, then obtaining more data is likely to help. | If the model is underfitting the training data, it has not captured the information in the examples you already have. Adding further examples will not help any more. |
| True       | The "error analysis" process of manually examining the examples which your algorithm got wrong can help suggest what are good steps to take (e.g., developing new features) to improve your algorithm's performance. | This process of error analysis is crucial in developing high performance learning systems, as the space of possible improvements to your system is very large, and it gives you direction about what to work on next. |
| True       | On skewed datasets (e.g., when there are more positive examples than negative examples), accuracy is not a good measure of performance and you should instead use F1 score based on the precision and recall. | **This is a wonderful interview question.**                  |
| True       | The error analysis process of manually examining the examples which your algorithm got wrong can help suggest what are good steps to take (e.g. developing new features) to improve your algorithm's performance | none needed                                                  |

















### IELTS

---

在判断题和heading题中不要单独用数字定位.

90%情况下定位词是名词. 10%动词.

1. 名词定位: 主语, 从句主语, 句末名词.

常见词(比如water)，文章主旨词(**标题词**)，题干内部重复出现的词(**小题之间重复的**)，不适合做定位词

**找到定位词去原文找**



NG: 如果题干说的很绝对, 只有... 原文中又出现但是不那么绝对, 就NG.

原文: 香蕉是已知的最古老的水果. 题干: 香蕉是已知的最古老的水果之一 (**选对**)

原文: 香蕉是已知的最古老的水果之一. 题干: 香蕉是已知的最古老的水果 (**选NG**)



**判断题数量规律**: Y > F > NG.

判断题需要同时定位多个题, 可以两个两个定位: 1, 2定位, 2, 3定位, 3, 4定位. 来防止NG出现, 保证在一题上不会花太多时间.

**判断题考点词分类**: be, can后面的通常是考点, 程度描述.