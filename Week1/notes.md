### Introduction
    易小鱼 2023 Summer
#### Basic concepts
##### Machine Learning? Looking for Function!
Diffrent tasks: Speech Recognition; Image Recognition; Playing games; etc.

##### Different types of Functions
Regression: outputs a scalar
Classification: given classes, outputs the correct one
Structured Learning: create something with structure

##### How to find a function?
  
  1. 假设一个model(with unknown parameters),based on domain knowledge
    eg. y=wx+b
  2. Define Loss(a function of parameters) from Training Data, which stands for how good a set of parameters is.
    eg. mean abs error/ mean square error, cross-entropy(for probability distribution)
  3. Optimization 梯度下降求参数 local min vs global min

##### Linear too simple! More sophisticated models needed!
Activation Functions
- sigmoid
   对于分段线性的函数，我们可以用许多function做线性组合得到！
   对于任意的曲线，取分划，线性逼近！

   对于水平-斜坡-水平的function,可以用sigmoid拟合。

   于是任意的分段线性的函数可以近似地写作
   $$y=b+\sum_ic_isigmoid(b_i+w_ix_i) $$ 
   thus getting more features rather than $y=b+wx$
   for $y=b+\sum_jw_jx_j$,we can also get
   $$y=b+\sum_ic_isigmoid(b_i+\sum_jw_{ij}x_i)$$
   
   We can write it in a more beautiful form via matrix and vector.

   $$y = b+\vec{c^T}\sigma(\vec b+W\vec{x})$$

   其中$W,\vec{b},c^T,b$为未知参数，把他们全部拉直写成一条长向量$\theta$.

   我们完成了第一步！find a function with unknown！
   接下来依旧一样，define loss，梯度下降
   在实际操作中，一共N组数据，分成L个batch，对每个batch分别算loss，更新参数；把所有的batch都看过一次称作epoch，每更新一次参数叫update.(eg. 10000 examples with batch size = 10, we have 1000 updates in 1 epoch)
- relu
  除了sigmoid，还有许多其他的拟合方式，如relu
##### We can repeat the process $y = b+\vec{c^T}\sigma(\vec b+W\vec{x})$
引出神经网络的概念，it got a fancy name Neural Network!
Many layers means Deep.--> Deep Learning.
Why "Deep" instead of "Fat" network?
How to select model?



  