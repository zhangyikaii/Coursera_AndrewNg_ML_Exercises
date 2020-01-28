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