
# mini-batch SGD 小批量随机梯度下降-分类

## 数据集来源

自行生成

$x_1^2 + x_2^2 - 1.5^2 = 0$内外随机产生1000个点用于训练，200个用于测试。 

欠拟合多项式：$h(\boldsymbol\theta)=\theta_1 + \theta_2x_1 + \theta_3x_2$

合适拟合多项式：$h(\boldsymbol\theta)=\theta_1 + \theta_2x_1 + \theta_3x_2 +\theta_4x_1^2 +\theta_5x_2^2$

过拟合多项式：$h(\boldsymbol\theta)=\theta_1 + \theta_2x_1 + \theta_3x_2 +\theta_4x_1^2 +\theta_5x_2^2 +\theta_6x_1^3 +\theta_7x_1^4$

## 拟合函数
采用多项式函数拟合，$h(\theta)=\theta_0+\theta_1x_1+\theta_2x_2+\cdots=\boldsymbol\theta^T\boldsymbol x$

## 损失函数-mini-batch学习，多个数据的交叉熵误差
### mini-batch学习
从全部数据中选出一部分，作为全部数据的“近似”。这种小批量数据叫做mini-batch，比如，从100000个训练数据中随机选择一批，如100个，用这100个训练数据的损失函数的总和作为学习指标，不断随机选择mini-batch，重复学习过程以更新参数实现参数寻优。
### 交叉熵的损失函数
此为二分类（是非）问题，模型最后需要预测的结果只有两种情况，**logistic**回归（是非问题）中，$y^{(i)}$为真实标签取0或者1
假设函数（hypothesis function）定义为：
$$h_{\theta}\left(\boldsymbol x^{(i)}\right)=\frac{1}{1+e^{-\boldsymbol\theta^{T} \boldsymbol x^{(i)}}}$$
其中$\boldsymbol x^{(i)}$为第$i$组数据，$\boldsymbol x^{(i)}=[1,x_1^{(i)},x_2^{(i)},x_3^{(i)},\cdots,x_n^{(i)}]^T$为$n+1$维向量  
则交叉熵损失函数为：
$$J(\theta)=-\frac{1}{m} \sum_{i=1}^{m} y^{(i)} \log \left(h_{\theta}\left(\boldsymbol x^{(i)}\right)\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(\boldsymbol x^{(i)}\right)\right)$$
- $m$是mini-batch的size,即有多少个数据，如$100$，除以$m$是为了正规化，求出单个数据的“平均损失函数”，以获得和训练数据数量无关的统一指标
- $y^{(i)}=1$时第二项$\left(1-y^{(i)}\right)=0$，$y^{(i)}=0$时，则第一项为0  

计算$J(\theta)$对第$j$个参数分量$\theta_j$求偏导,得到交叉熵对参数的导数:
$$\frac{\partial}{\partial \theta_{j}} J(\theta)=\frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(\boldsymbol x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}$$
目的为求得最小的$J(\theta)$：  
重复{  
$$\theta_{j}:=\theta_{j}-\alpha \frac{\partial}{\partial \theta_{j}} J(\theta)$$  
同时对所有的$\theta_j$更新，其中$j=1,2,3,\cdots,n$  
}  
代入偏导，即为：  
重复{  
$$\theta_{j}:=\theta_{j}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(\boldsymbol x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}$$  
同时对所有的$\theta_j$更新，其中$j=1,2,3,\cdots,n$，$\alpha$为步长   
}  
可以看出，$\theta_{j}$更新和线性回归中梯度下降算法的$\theta_{j}$更新一致，差别的是假设函数$h_{\theta}$不同。
## 运行结果

# 欠拟合
![训练损失](https://github.com/BillowRock/PatternRecognition/raw/master/minibatchSGD/underfit_loss.png)  
![欠拟合](https://github.com/BillowRock/PatternRecognition/raw/master/minibatchSGD/underfit_circle.png)  

可以看出在欠拟合情况下无法进行拟合，训练后对测试集分类正确率仅有50%左右，基本没效果

# 合理拟合
![训练损失](https://github.com/BillowRock/PatternRecognition/raw/master/minibatchSGD/fit_loss.png)  
![合理拟合](https://github.com/BillowRock/PatternRecognition/raw/master/minibatchSGD/fit_circle.png)  

选取正确多项式，拟合结果正常

# 过拟合
![训练损失](https://github.com/BillowRock/PatternRecognition/raw/master/minibatchSGD/overfit_loss.png)  
![过拟合](https://github.com/BillowRock/PatternRecognition/raw/master/minibatchSGD/overfit_circle.png)  

可看出分类曲线出现了对数据集的迎合，但因容量增加有限，与合理拟合效果相差不大
