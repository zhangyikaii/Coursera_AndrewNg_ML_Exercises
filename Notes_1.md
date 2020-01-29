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