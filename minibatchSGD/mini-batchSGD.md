# mini-batch SGD 小批量随机梯度下降

## 数据集来源

[皮肤数据集](http://archive.ics.uci.edu/ml/datasets/Skin+Segmentation)，前三列为B，G，R（x1，x2和x3特征）值，第四列为标签（决策变量值y），将其中70%划分为训练集，30%划分为测试集。
## 拟合函数
采用多项式函数拟合，$h(\theta)=\theta_0+\theta_1x_1+\theta_2x_2+\theta_3x_3=\boldsymbol\theta^T\boldsymbol x$
## 特征归一化
$x_i=(x_i-min)/(max-min)$，对于表示RGB格式，其中$min=0$，$max-min=255$，将$[0,255]$转化为$[0,1]$
## 损失函数-mini-batch学习，多个数据的交叉熵误差
### mini-batch学习
从全部数据中选出一部分，作为全部数据的“近似”。这种小批量数据叫做mini-batch，比如，从100000个训练数据中随机选择一批，如100个，用这100个训练数据的损失函数的总和作为学习指标，不断随机选择mini-batch，重复学习过程以更新参数实现参数寻优。
### 交叉熵的损失函数
此为二分类（是非）问题，模型最后需要预测的结果只有两种情况，**logistic**回归（是非问题）中，$y_i$为真实标签取0或者1
假设函数（hypothesis function）定义为：
$$h_{\theta}\left(x^{(i)}\right)=\frac{1}{1+e^{-\boldsymbol\theta^{T} x^{(i)}}}$$
其中$\boldsymbol x^{(i)}$为第$i$组数据，$\boldsymbol x^{(i)}=[1,x_1^{(i)},x_2^{(i)},x_3^{(i)}]^T$为$3+1$维向量  
则交叉熵损失函数为：
$$J(\theta)=-\frac{1}{m} \sum_{i=1}^{m} y_i \log \left(h_{\theta}\left(x^{(i)}\right)\right)+\left(1-y_i\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)$$
- $m$是mini-batch的size,即有多少个数据，如$100$，除以$m$是为了正规化，求出单个数据的“平均损失函数”，以获得和训练数据数量无关的统一指标
- $y_i=1$时第二项$\left(1-y_i\right)=0$，$y_i=0$时，则第一项为0  

计算$J(\theta)$对第$j$个参数分量$\theta_j$求偏导,得到交叉熵对参数的导数:
$$\frac{\partial}{\partial \theta_{j}} J(\theta)=\frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y_i\right) x_{j}^{(i)}$$
