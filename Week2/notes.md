### What to do if fails to train
      易小鱼 2023 Summer
#### Review
1. function with unknown $\theta,y=f_\theta(x)$
2. $L(\theta)$
3. $\theta^*=\argmin_\theta L$

#### General guide on training
Some problems that might happen
##### 当Loss在Training Data上不够低时
1. model bias
   模型本身不够复杂
2. Optimization Issue
   局部最小值
如何辨明是1还是2？ 可以先从比较简单的模型开始尝试，比如线性模型，因为他们比较容易optimize。并且一个最基本的事实是，更复杂的模型应该能达到比简单模型更好的结果。基于这个事实去判断究竟是不够复杂还是优化没做好。

##### 当训练集上误差小但是在测试集上误差大
1. 可能是过拟合Overfitting!
    Why? An extreme example as follows:
    $$
    f(x)=\begin{cases}
        y_i & \text{if } \exist x_i=x,\\
        random & \text{if } others,
        \end{cases}      
    $$
    How to solve?
    - more training data
    - data augmentation
    - constrained model(less paras, sharing paras, less features, early stopping, regularization, dropout)
2. mismatch
   训练资料跟测试资料的distribution是不一样的，此时增加训练资料也不会有帮助。

#### 怎么样选合适的model
过于复杂，导致overfitting, 不够复杂，导致model bias.
How to choose a proper one?
In other words, how do we evaluate the performance of a certain model?
使用Cross Validation:
在Training Set上训练模型，在Validation Set上验证表现
怎样分不同的Set? K折交叉验证！

#### gradient = 0 卡在local minima 或 鞍点(saddle point)怎么办
   local minima and saddle point统称为critical point
   那遇到critical point时如何知道是局部最小还是鞍点？
   用泰勒展式去逼近，事实上
   $$
   L(\theta)\approx L(\theta^{'})+(\theta-\theta^{'})^Tg+\frac{1}{2}(\theta-\theta^{'})^TH(\theta-\theta^{'})
   $$
   其中
   $$
   g \text{ for gradient, }g=\nabla L(\theta^{'})\\
   H \text{ for hessian matrix, }H_{ij}=\frac{\partial^2}{\partial\theta_i\partial\theta_j}L(\theta^{'})
   $$
   当gradient=0时，判断其黑塞矩阵，若正定，局部最小；若负定，局部最大；若不定，鞍点！（对于实对称矩阵，用特征值判断是很好的方法）

   如果发现是遇到鞍点，黑塞矩阵会给出优化的方向，去找其负特征值$\lambda$对应的特征向量$\alpha$，那么让$\theta \leftarrow \theta+\alpha,\Rightarrow L(\theta) \leftarrow L(\theta)+\frac{1}{2}\lambda\alpha^T\alpha$，从而$L(\theta)$会变小！

   不过事实上，这种方法实践上比较少用。

   问题：低维的local minima放到高维会不会就是saddle point？进而有办法优化？

   在实践上发现，其实卡在local minima的情况较少，大部分情况下训练会卡在鞍点处。


#### Batch有什么用？
概念回顾：batch,epoch,update,shuffle
Shuffle after each epoch.
- Full batch v.s. batch size=1
  long time but powerful/short time but noisy
  不过在平行计算的条件下，大的batch不一定需要更长的时间，除非batch size真的非常大

在平行运算的条件下，大的batch size在跑完一个epoch上占有时间优势。
However, "Noisy" update is better for training, thus smaller batch size has better performance.
理解：假设选用MSE为损失函数
$$
L(\theta)=||(f(x,\theta)-y)||,f \text{ for function},x,y\in \R^n
$$
在full batch情况下，x与y全部都是给定的，于是L是一个给定的与$\theta$有关的函数，可能会卡住！
但当batch size比较小时,每个batch都对应一个不同的$L(\theta)$，所以在某个batch时卡住换一个不一定会卡住。

在测试时，小的batch相较于大的batch，由于其参数方向更新更为noisy，倾向于收敛到flat minima，而大的batch倾向于收敛到sharp minima.于是当train data与testing data存在差距时，flat minima相较于sharp minima更能抵抗住差距带来的变化。

#### Momentum技术
想法：把损失函数超斜坡视作真实物理世界，把损失函数的下降形象地看成小球滚下坡，于是当遇到鞍点/局部最小点时，由于小球存在惯性/动量，小球不一定会被卡住！
Movement: delta = hpp1 * movement of last step -  hpp2 * gradient at present(hpp for hyperpara)
事实上，上一步的移动也可以由之前所有梯度的加权和得到，于是这种方法还可以解读为考虑过去所有的梯度而非考虑当次的梯度。

#### 总结
critical points have zero gradients
consider Hessian matrix
if saddle points, escape along the direction of eigenvectors.
local minima can be rare.
smaller batch and momentum might help escape critical points.

#### How to adjust learning rate automatically?
学习率过大可能震荡，过小可能不收敛，有没有办法让学习率能够自动调整？
想法：梯度陡峭时学习率小一些，梯度平坦时学习率大一些
一些记号的约定：
$$
\vec{\theta}=(\theta_1,\cdots,\theta_n)^T\in \R^n\\
\eta=\text{learning rate}\\
\vec{g}=(g_1,\cdots,g_n)^T\in\R^n,g_i=\frac{\partial L}{\partial\theta_i}\\
t \text{ for iteration times}
$$
对$\theta$的第i个分量$\theta_i$做优化时，之前的做法是
$$
\theta_i^{t+1}=\theta_i^t-\eta g_i^t
$$
现在考虑添加一项$\sigma_i$来调整学习的速度，我们有
$$
\theta_i^{t+1}=\theta_i^t-\frac{\eta}{\sigma_i^t} g_i^t\quad,\quad \sigma_i^t=\sqrt{\frac{1}{t+1}\sum_{i=0}^t(g_i^t)^2}
$$
通过这个办法(Root Mean Square)来达到动态地调整每个方向上学习的速率。
然而此方法平等地对待每次的梯度，真正学习时可能会对第t次的梯度赋予更多的关注，于是进行改进，得到RMSProp方法，将$\sigma_i^t$修改如下
$$
\sigma_i^t=\sqrt{\alpha(\sigma_i^{t-1})^2+(1-\alpha)(g_i^t)^2},\alpha\in(0,1)
$$
通过修改$\alpha$来控制以往的梯度与最新的梯度之间的权重。
在实践中，常用Adam方法，即RMSProp + Momentum
$$
\theta_i^{t+1}=\theta_i^t-\frac{\eta}{\sigma_i^t} m_i^t,\quad m_i^t\text{ for Momentum}
$$
注意：RMSProp考虑的是陡峭的绝对程度，关注大小，而Momentum是直接将过去的梯度带符号加权求和，关注大小和方向，二者不会相互抵消。

还有一种常用的做法，让学习率随着时间变化
- learning rate decay
  让学习率随着迭代次数缓慢下降（认为训练到后面已经接近结尾，不需要太大的学习率）
- warm up
  学习率随着时间先增大后减小

#### 直到目前为止，考虑的都是在一个复杂的error surface上如何做优化，但有没有办法直接将error surface本身变简单？
一个简单的例子：$y=w_1x_1+w_2x_2+b$，如果$x_1,x_2$的取值范围相差很远，就会造成$w_1,w_2$两个维度上error surface的崎岖程度相差很大。换言之，我们希望input的每一个维度的数据分布都相似。

#### Batch Normalization
假设batch size = n，也就是一次输入n个向量$\vec{x_1},\cdots,\vec{x_n}$，考虑第i维的分量$x_{1i},\ldots,x_{ni}$，平均值为$m_i$，标准差为$\sigma_i$，那么计算$\hat{x_{ji}}=\frac{x_{ji}-m_i}{\sigma_i},j=1,\ldots,n$,就会得到一组新的以0为平均值，1为方差的数据。
在神经网络中，我们可以任意地对每一层的输入，输出选择做normalization与否。

需要注意，做了Normalization以后，平均值和方差与上一层的所有变量都相关，于是后续的神经网络将会变得庞大！

在实践中，完成normalization得到$\vec{\hat{x_i}}$以后，会考虑$\gamma\odot\vec{\hat{x_i}}+\beta$,其中$\odot$表示逐元素相乘。初始令$\gamma=\vec{1},\beta=\vec{0}$,并让机器学习。这样的目的是为了让数据平均值不是0，减小对神经网络的限制。

在测试时，不是一个一个batch送进来！那还何来平均值与方差？
Computing the moving average of 平均值 and 方差 during training as the 平均值 and 方差 when testing。


#### 分类问题也是回归问题？
通过sigmoid函数、softmax函数实现分类问题。
损失函数为何交叉熵好过MSE?
1. 交叉熵等价于最大似然
2. 在optimization上CrossEntropy具有优势！（数学上可以给出证明）
上面的例子也说明不同损失函数的选择会影响训练的难度！


#### 如何选一个好的训练集？
一些记号的约定：
$$
h\in H,\text{parameters to be optimized}\\
D,\text{Data set}\\
L(h,D),\text{Loss function}
$$

一般来说，我们无法获取数据的全集$D_{all}$,我们也不会使用数据全集进行训练，我们会取一部分子集$D_{train}\subset D_{all}$作为训练集。然而，训练集的分布与全集的分布类似时，训练出的参数才会具有较好的泛用性，那么如何衡量训练集能否很好地代表全集呢？

衡量训练集能否很好地代表全集的标准：
$$
\forall h\in H,|L(h,D_{train})-L(h,D_{all})|\le\epsilon
$$
即对参数的所有取值可能，如果训练集与全集的误差都相近，那么就可以认为这个训练集可以较好地代表全集。

相应地，若
$$
\exist h, s.t. |L(h,D_{train})-L(h,D_{all})|>\epsilon
$$
就认为这个训练集是不好的。

直观上，增加训练资料的数目，或者缩减H的取值范围，可以减小取到坏训练集的概率。

