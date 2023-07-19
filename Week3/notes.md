### Convolutional Neural Network(CNN)
### Image as input!
      易小鱼 2023 Summer
#### Let's start from fully connected network.
假设有一张$100*100$的图片作为神经网络的输入，考虑其RGB三通道，可以将其拉直为$100*100*3$的向量来输入。假设hidden layer1有1000个neuron，那么input->hidden layer1就有$3*10^7$个weight.

真的需要这么多神经元吗？$100*100*3$表示了每个通道的每个像素，真的需要每个像素都给予关注吗？

#### 事实1：识别图像时我们更关心图像的某些pattern
每个neuron只关注一个小区域Receptive field,而不是关注整张图片。比如只关注一个$3*3$的小区域，那么就只对应$3*3*3$个weight（考虑三个通道）

事实上，Receptive field的尺寸，形状，是否三通道都关注完全取决于对问题的理解。

不过常见来说一般三个通道都会关注，因此不再特别地强调RGB三个不同的通道。把Receptive field的尺寸成为kernal size,如$3*3$，把Receptive field移动的步长称为stride,若Receptive field移动到图像的边缘，需要将边缘补齐的情况称为padding。padding时可以补0，或者补某种可能的平均值，或者补其他合理的数值，取决于对问题的理解。

同一个Receptive field不一定只有一个neuron在侦测，比如在辨识鸟的任务上，一个Receptive field可能同时被侦测鸟嘴，翅膀，鸟爪的neuron所覆盖。

#### 事实2：同一个pattern在不同图片可能位置不一样
在不同鸟的图片中，鸟嘴可能出现在不同的区域，若如上文所述每一个Receptive field都安排多个neuron，参数量可能太多了！而且在不同的地方做的事情是一样的，都是侦测鸟嘴！

让不同感受野(Receptive field)的神经元共享参数！
Each receptive field has the neurons with the same set of parameters(called filters)

换言之，每个filter去图片里面抓取某个pattern,通过按stride遍历所有receptive field，然后得到一张新的图片，称之为feature map.

有多少个filter,就能得到多少channel的feature map.
#### 总结以上两点
全连接->感受野->参数共享 网络的弹性是在减小的！
把receptive field+parameter sharing称为convolutional layer! NN with convolutional layer is called CNN.

#### 多层卷积层
在每层卷积层，filter的高度应该是输入的channel数。
假设固定是$3*3$的filter，那么第一层卷积层看的就是原图中$3*3$的部分，但是第二个卷积层看的是第一个卷积层中的$3*3$，对应原图中$5*5$的部分。因此通过层层卷积，随着网络变深，小特征聚合成大特征，不用担心看不到大范围的feature.

#### 事实3：subsampling不会改变图像的内容
例如，对于一个图片，去掉偶数行和奇数列，并不会改变这个图片里面是什么东西。
引入pooling的概念，例如：maxpooling，取每个pool中最大值作为代表，如一张$4*4$的图片，划分成4个$2*2$的小块，经过maxpooling以后得到一个$2*2$的新图像。
不见得一定是maxpooling,有很多pooling的方法。不难看见，经过pooling后，把图像变狭长了($4*4*64->2*2*64$)

#### Whole CNN Network
input->convolution->pooling(可能交替几次)->flatten(将pooling得到的向量拉直为一个长向量)->全连接

#### 什么样的问题适合CNN？
- 需要关注图像某个局部区域的小pattern
- 同样的pattern会出现在不同的地方
所以CNN也可以用来下围棋！
不过在下围棋这个问题中，做pooling对结果会产生影响，所以alpha go没有用pooling！

通过filter的特性，可以合理猜测CNN不具有缩放、旋转不变性。如何解决？通过加一层Spatial Transformer Layer来解决！
#### Spatial Transformer Layer
通过加一层先把image进行空间上的变换。但是如何实现呢？
规定好平面坐标系，通过线性代数上的一些坐标变换！（6个参数；4个旋转，2个平移）
ST不仅对image，还能对feature map进行空间变换。

原理：
输入一张图片，通过localisation net获得矩阵参数$\theta$,然后通过坐标变换的公式，结合$\theta$与输入图片一起，得到输出图片（注意在这个过程中遇到非整数的坐标可用双线性插值的办法解决，目的是确保可微，从而才能用梯度下降）。