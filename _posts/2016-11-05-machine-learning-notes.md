---
layout: post
date:   2016-11-05 21:00:00 +0800
---

交大CS课程 人工智能 复习笔记。由于是复习时所做，较为粗糙，待有机会再维护。

(按本届的情况，不建议选这门课 =.=)。

参考材料：Andrew ng旧版standford机器学习公开课

跳过了
- SMO算法(第8集)
- 经验风险最小化，特征选择，贝叶斯统计正则化(9-11集)
- 因子分析（13集）
- ICA（14集）
- 16-20集

# 0. 矩阵运算
矩阵运算求导时常用的公式

$$
trAB=trBA\\

trABC=trCAB=trBCA\\

A=(a_{ij})_{m\times n},\ \nabla_A=(\frac{\partial}{\partial a_{ij}})_{m\times n}\\

\nabla_AtrAB=B^T\\

trA=trA^T\\

\nabla_AtrABA^TC=CAB+C^TAB^T
$$

当A是一个标量时，有$trA=A$

用例：

$$
\nabla_uu^Tu=\nabla_utr\ u^Tu=\nabla_utr\ uu^T=\nabla_utr\ uEu^TE=EuE+E^TuE^T=2u
$$


# 1. 最小二乘法


$$
h_\theta(x)=\theta^Tx\\

min_{\theta}(\frac{1}{2}\sum_{i=0}^{n}(h_{\theta}(x)-y)^2)=min(J(\theta))\\


\theta_{i}^{k+1} = \theta_{i}^{k} - \alpha\frac{\partial}{\partial\theta_{i}}J(\theta)\\

\frac{\partial}{\partial\theta_{i}}J(\theta)=\sum_j(h_{\theta}(x^j)-y^j))x_i^j\\

or\\

\theta = (X^TX)^{-1}X^TY
$$

最小二乘法等价于以下假设的最大似然估计

$$
y=\theta^Tx+\epsilon\\

\epsilon\sim N(0,\sigma)
$$


即最小化loss函数等价于最大化以上分布的似然

# 2. PCA
目的：得到数据方差最大的投影方向u

预处理
- 减去平均值
- 除以方差
- 


$$
max_{\vert u\vert =1}\frac{1}{m}\sum_i (x_i^Tu)^2=\frac{1}{m}\sum_i (u^Tx_i)(x_i^Tu)\\

=u^T(\frac{1}{m}\sum_i (x_ix_i^T))u=u^T\Sigma u
$$


运用拉格朗日乘数法

$$
L(u,\lambda)=u^T\Sigma u-\lambda(uTu-1)\\

\nabla_uL=\Sigma u-\lambda u=0
$$

即u是sigma的特征向量

## 高维向量的PCA实现 SVD
当数据的维度n很高时，sigma矩阵的大小n^2，通常是难以处理的。这时候我们考虑将sigma矩阵进行奇异值分解(SVD)来解决特征值问题。一个nxn的矩阵A总可以分解为A=UDV^T，其中U，D和V都是nxn的矩阵，而D是对角阵。

如果A是mxn的矩阵，可有这样的分解

$$
A_{m\times n}=U_{m\times n}D_{n\times n}V^T_{n\times n}
$$

其中D是对角阵，并且U中的向量相互正交，V中也是一样。而U的列向量将是AA^T 的特征向量，V的列向量将是A^TA的特征向量。

由于sigma矩阵=X^TX(X的每一行是一条数据),故我们只需要计算出X矩阵的SVD分解，即可得到sigma的特征向量V的列向量和相应的特征值D。

# 3. logistic regression
假设符合以下形式的分布

$$
y\in\{0, 1\}\\

p(y\vert x)=h(x)^y(1-h(x))^{1-y}\\

h(x)=\frac{1}{1+e^{-\theta^Tx}}
$$

目的是求theta使得p(y\vert x)是数据的最大似然估计

$$
J(\theta)=\Pi p(y^j\vert x^j)\\

min_\theta\ j(\theta)=log(J(\theta))
$$

迭代算法

$$
\theta_{i}^{k+1} = \theta_{i}^{k} - \alpha\frac{\partial}{\partial\theta_{i}}j(\theta)\\

\frac{\partial}{\partial\theta_{i}}j(\theta)=\sum_j(h_{\theta}(x^j)-y^j))x_i^j
$$


p(x\vert y)符合高斯分布=>p(y\vert x)符合logistic分布，但反向不成立

实际上，只要p(x\vert y)符合任何一个指数分布族，都有p(y\vert x)符合logistic分布

# 4. FLDA(Fisher linear discriminant analysis)
将数据线性投影到一维，然后设置一个阀值将数据分开。通过最大化离散度实现。

$m_i=E(w^Tx_j), x_j\in i$

类内离散度$s_i=\sum_{x_j\in i}(w^Tx_j-E(w^T-x_j))^2$

类间离散度$s_b=(m_1-m_2)^2$

最优化

$$
max\frac{s_b}{s_1+s_2}
$$


等价于

$$
max\ w^TS_bw\\

s.t. w^TS_ww=c\neq0\\

S_b=(m_1-m_2)(m_1-m_2)^T\\

S_i=\sum_{x_j\in i}(x_j-m_i)(x_j-m_i)_T\\

S_w=S_1+S_2
$$


拉格朗日乘数法

$$
L=w^TS_bw-\lambda(w^TS_ww-c)\\

\frac{\partial L}{\partial w}=0\\

\Rightarrow S_bw-\lambda S_ww=0\\

w=\frac{1}{\lambda}S_w^{-1}(m_1-m_2)(m_1-m_2)^Tw
$$

把后面两个向量乘起来得到w的方向为$\frac{1}{\lambda}S_w^{-1}(m_1-m_2)$

# 5. SVM
函数间隔(function margin)

$$
y_i\in\{-1,+1\}\\

\hat{\gamma_i}=y_i(w^Tx_i+b)
$$

几何间隔(geomatry margin)

$$

\gamma_i=\frac{\hat{\gamma_i}}{\vert w\vert }

$$

函数间隔或几何间隔大于0的时候我们知道分类是正确的。

最优间隔分类，即使找出决策面w，使得到决策面最近的点 到决策面的距离最大，并且每个数据的分类都正确。

$$

max_{w,b}(min_i(\gamma_i))

s.t.\ \gamma_i>0\ for \ each\ i


$$

上述问题在数学上存在着一定的困难，我们需要考虑转化问题，使得问题可解。由于w可以任意缩放而不影响几何间隔，所以我们可以通过对\vert w\vert 增加约束来简化问题。首先上面的问题可以等价于

$$

max_{\hat{\gamma},w,b}(\frac{\hat{\gamma}}{\vert w\vert })

s.t.\ \hat{\gamma}_i\ge\hat{\gamma}\ for\ each\ i

$$

即找到上述最优的决策面w，和其对应的最大的 最小函数间隔。实际上在这一步看起来是把问题变复杂了，因为引入了新的需要优化的变量gamma hat。

接下来在这个基础上增加约束gamma hat = 1，使得问题变为

$$

max_{w,b}(\frac{1}{\vert w\vert })

s.t.\ \hat{\gamma}_i\ge1\ for\ each\ i

$$

注意，在此我们只是将最小函数间隔定为1，这其实是间接给\vert w\vert 增加了约束，使得在几何间隔最小的点处，函数间隔恰好是1。

更进一步，上面的问题当然等价于

$$

min_{w,b}(\frac{1}{2}\vert w\vert ^2)

s.t.\ \hat{\gamma}_i\ge1\ for\ each\ i

$$

这个问题是一个凸优化问题，有着比较好的性质

对于不等式约束g，可以用一般化的lagrange乘数法，将上述问题转化为

$$

g_i(w)=\hat{\gamma_i}(w)-1

L(w,\alpha)=\frac{1}{2}\vert w\vert ^2-\sum_i\alpha_ig_i(w)

\theta(w)=max_{\alpha\ge0}L(w,\alpha)

p^\star=min_{w,b}max_{\alpha\ge0}L(w,\alpha)

$$

当原约束条件有不满足时，会有theta=无穷大。而当原约束条件都满足时，有theta=\vert w\vert ^2 ，即我们要最小化的值。这样，在我们进行p^star中min优化的时候，就要求w满足我们的约束条件，这样才能取到最小。

故原优化问题可以转化为对p^star的优化。

进一步地，我们将p^star的优化转化为对偶问题

$$

d^\star=max_{\alpha\ge0}min_{w,b}L(w,\alpha)

$$

这里有

$$

d^\star\le p^\star

$$

在满足如下KKT条件时，两者相等

$$

\exists w^\star,\alpha^\star that

\forall i\ g_i(w^\star)\ge0

\alpha_i^\star\ge0

\alpha_i^\star g_i(w^\star)=0

\nabla_wL(w^\star)=0

$$


这保证了

$$

\forall i\ \alpha_i^\star=0\ or\ g_i(w^\star)=0

$$


d^star中，利用KTT条件，通过w对L进行最小化后，可以得到（推导较复杂）

$$

w=\sum_i\alpha_iy_ix_i

\sum_i\alpha_iy_i=0


L(w,\alpha)=\sum_i\alpha_i-\frac{1}{2}\sum_{i,j}\alpha_i\alpha_jy_iy_jx_i^Tx_j

$$

此时只需通过alpha最大化L即可得到w和b的解，问题变为

$$

max_\alpha \sum_i\alpha_i-\frac{1}{2}\sum_{i,j}\alpha_i\alpha_jy_iy_jx_i^Tx_j

s.t.\ \alpha_i\ge0\ and\ \sum_i\alpha_iy_i=0

$$

这个问题寻找的alpha可以通过SMO算法解决。

对于有噪声的数据，我们还要加入松弛变量，问题变为(约束相当于多了n个gi)

$$

min_{w}(\frac{1}{2}\vert w\vert ^2+C\sum_i\xi_i)

s.t.\ \hat{\gamma}_i\ge1-\xi_i\ and\ \xi_i\ge0\ for\ each\ i

$$

此时最终的优化问题变为

$$

max_\alpha \sum_i\alpha_i-\frac{1}{2}\sum_{i,j}\alpha_i\alpha_jy_iy_jx_i^Tx_j

s.t.\ 0\le\alpha_i\le C\ and\ \sum_i\alpha_iy_i=0

$$

与上面类似

由以上算法得到svm的分类函数为

$$

w=\sum_i\alpha_iy_ix_i

b=-\frac{1}{2}(max_{i,y_i=-1}w^Tx_i+min_{i,y_i=1}w^Tx_i)

h(x)=w^Tx+b=\sum_i\alpha_iy_i(x_i,x)+b

$$

是一些内积的和的形式，根据h(x)的正负号可以将x分类。

由此，可以引入核函数，将x和xi映射到高维空间后的内积快速计算。对于高维映射phi，相应的核函数K作用如下

$$

K(x_i,x)=(\phi(x_i),\phi(x))

h(x)=w^Tx+b=\sum_i\alpha_iy_i(\phi(x_i),\phi(x))+b

$$

常用的将向量映射到无穷维的高斯核函数为

$$

K(x_i,x)=exp(-\frac{\vert x_i-x\vert ^2}{2\sigma})

$$

多项式核

$$

K(x_i,x)=(x_i^Tx)^d

$$


# 6. k-means
1. 随机选择k个重心
2. 将每个数据xi标记为离它最近的一个重心j类，若没有数据的标记改变，则结束
3. 重新根据各个xi的分类情况计算k个重心，重复2

# 7. mixture gaussian
## EM算法 最大期望算法
考虑分布p(x,z;theta)，只知道{x_i}的数据，希望估计参数theta使得

$$

max_{\theta}\ l(\theta)=\sum_i log\ p(x_i;\theta)=\sum_i log \sum_j p(x_i,z_i=j;\theta)

$$

由对数函数的凸性，有

$$

max_{\theta}\ l(\theta)=\sum_i log\ p(x_i;\theta)=\sum_i log \sum_j p(x_i,z_i=j;\theta)\\

=\sum_i log \sum_j Q_i(z_i=j)\frac{p(x_i,z_i=j;\theta)}{Q_i(z_i=j)}\\

\ge\sum_i  \sum_j Q_i(z_i=j)log \frac{p(x_i,z_i=j;\theta)}{Q_i(z_i=j)}\ (\star)\\

for\ Q_i(z_i=j)\ge0\ and\ \sum_j Q_i(z_i=j)=1

$$


由此找到了似然函数的一个下界函数。下面来求Q_i，来让上式取等。注意，上述不等式对任意theta都成立，而此处我们将theta看作常数，即上一次迭代中得到的theta，我们将上式取等以后，将得到一个当前theta下以上形式的最高的下界函数。此时我们再优化theta，将得到一个使得似然函数更大的theta，同时又保证不等式仍然成立。

由Jensen不等式取等的条件，我们有

$$

\forall i,\frac{p(x_i,z_i=j;\theta)}{Q_i(z_i=j)}=constant

$$

即

$$

Q_i(z_i=j)=\frac{p(x_i,z_i=j;\theta)}{\sum_k p(x_i,z_i=k;\theta)}

=\frac{p(x_i,z_i=j;\theta)}{p(x_i;\theta)}

=p(z_i=j\vert x_i;\theta)

$$


由此，我们通过最大化(*)来得到l(theta)在当前最大似然估计

EM算法即先猜测z_i和theta，然后
1. E-step 计算

$$

Q_i(z_i=j)=p(z_i=j\vert x_i;\theta)

$$

2. M-step 利用1的Q，通过取到最大似然下界的最大值，重新计算theta。注意，此处Q看作常数，而theta看作变量。

$$

\theta'=argmax_\theta\ \sum_i  \sum_j Q_i(z_i=j)log \frac{p(x_i,z_i=j;\theta)}{Q_i(z_i=j)}

$$


EM的另一种理解是，在E步中，我们通过Q取(*)式的极大值，而在M步中，我们通过theta取似然函数l(theta)的极大值，这样每一步中，概率都在上升，这将使得似然函数的下界逐步提高，达到一个局部最大值。

EM算法能起效的主要原因是，找到原始似然函数导数的零点是困难的，但找到其下界函数导数的零点是容易的。

## 高斯混合模型
考虑多个高斯分布叠加在一起的分布。如果我们知道每个数据点(x_i)属于哪一个高斯分布(z_i=j)，那利用最大似然估计，我们可以得到每个高斯分布的参数，即均值(mu_j)，协方差矩阵(sigma_j)和属于该分布的概率phi_j。

而当我们不知道每个数据点对应的分布的时候，我们可以猜测每个数据点属于的高斯分布，然后利用猜测的结果对高斯分布的参数进行估计，再通过对参数的估计改进对每个数据点所属分布的判断。

MG的EM算法步骤
1. E-step 猜测z_i，即计算w^i_j=p(z_i=j\vert x_i; mu, sigma, phi) (利用贝叶斯公式)

$$

w_i^j=p(z_i=j\vert x_i; \mu, \sigma, \phi)

= \frac{p(x_i\vert z_i=j)p(z_i)}{\sum_k p(x_i\vert z_i=k)p(z_i=k)}

$$

2. M-step 重新估计参数，即利用E-step得到的z_i估计，通过极大似然估计和贝叶斯公式重新计算各高斯分布的参数miu, sigma, phi

$$

\phi_j=p(z=j)=\frac{1}{m}\sum_i w_i^j

\mu_j=\frac{\sum_i w_i^j x_i}{\sum_i w_i^j}

\Sigma_j=\frac{\sum_i w_i^j (x_i-\mu_j)(x_i-\mu_j)^T}{\sum_i w_i^j}

$$


# 8. HMM
假设

$$

p(x_i\vert x_{i-1},x...,o...)=p(x_i\vert x_{i-1})\\

p(x_{i+1}\vert x_i)=p(x_{j+1}\vert x_j)\forall i,j\\

p(o_t\vert x_t,x...,o...)=p(o_t\vert x_t)

$$

其中x_i是i时刻隐藏状态，o_i是i时刻观测到的状态。

参数

$$

b_{ij}=p(o_t=j\vert x_t=i)\\

a_{ij}=p(x_t=j\vert x_{t-1}=i)

$$

1. 已知参数，求给定观测序列的概率，动态规划

$$

\alpha_{t+1}(j)=p(x_{t+1}=j,o_1...o_{t+1})=p(o_1...o_{t+1}\vert x_{t+1}=j)p(x_{t+1}=j)\\
 
=p(o_{t+1}\vert x_{t+1}=j)p(o_1...o_{t}\vert x_{t+1}=j)p(x_{t+1}=j)\\

=p(o_{t+1}\vert x_{t+1}=j)p(o_1...o_{t},x_{t+1}=j)\\

=p(o_{t+1}\vert x_{t+1}=j)\sum_ip(o_1...o_{t},x_t=i,x_{t+1}=j)\\

=p(o_{t+1}\vert x_{t+1}=j)\sum_ip(o_1...o_{t},x_t=i)p(x_{t+1}=j\vert x_t=i,o_1...o_t)\\

=p(o_{t+1}\vert x_{t+1}=j)\sum_ip(o_1...o_{t},x_t=i)p(x_{t+1}=j\vert x_t=i)\\

\alpha_{t+1}(j)=b_{jo_t}\sum_i \alpha_t(i)a_{ij}\\

p(o_1...o_T)=\sum_j a_T(j)

$$

2. 已知参数和观测序列，求最可能隐藏序列

$$

pr(x_1...x_t=j\vert o_1...o_t)=max_{x_{1..t-1}}\ p(x_1...x_t=j\vert o_1...o_t)\\

=max_i\ pr(x_1...x_{t-1}=i\vert o_1..o_{t-1})p(x_t=j\vert x_{t-1}=i)p(o_t\vert x_t=j)

$$

计算pr的同时，记录取最大值时从t-1出发的路径即可。
3. 已知观测序列和隐藏状态，求参数(EM算法) 这里推导好像有点问题

$$

E-step\\

(1)\ \beta_t(j)=p(o_{t+1}...o_T\vert x_t=j)\\

=\sum_i p(o_{t+2}...o_T\vert x_{t+1}=i)p(o_{t+1}\vert x_{t+1}=i)p(x_{t+1}=i\vert x_t=j)\\

=\sum_i \beta_{t+1}(i)b_{io_{t+1}}a_{ji}\\

\beta_T(i)=1\\

(2)\ \xi_t(i,j)=p(x_{t+1}=j,x_t=i\vert O)\\

=\frac{p(x_{t+1}=j, x_t=i, O)}{p(O)}\\

=\frac{\alpha_t(i)a_{ij}\beta_{t+1}(j)b_{jo_{t+1}}}{\sum_i \sum_j \alpha_t(i)a_{ij}\beta_{t+1}(j)b_{jo_{t+1}}}\\

(3)\ \gamma_t(i)=p(x_t=i\vert O)\\

=\frac{\alpha_t(i)\beta_t(i)}{\sum_i\alpha_t(i)\beta_t(i)}\\

=\sum_j\xi_t(i,j)\\

M-step\\

(4)update\\

\pi'(i)=\gamma_1(i)\\

a_{ij}'=\frac{\sum_t\xi_t(i,j)}{\sum_t\gamma_t(i)}\\

b_{ij}'=\frac{\sum_{t,o_t=j}\gamma_t(i)}{\sum_{t}\gamma_t(i)}

$$




# 9. BPN

# 10. naive bayes
假设和公式

$$

p(x\vert y)\sim N\\

p(xy)=p(x\vert y)p(y)=p(y\vert x)p(x)\\

p(x)=\sum_i p(x\vert y_i)p(y_i)

$$

naive bayes算法的内容是从训练数据中找到p(x\vert y)和p(y)的分布，从而给测试数据分类，即求p(y_i\vert x)

$$

find\ p(x\vert y)\ and\ p(y)\ so\\

p(y_i\vert x) = \frac{p(x\vert y_i)p(y_i)}{p(x)}

$$

对$p(x\vert y_i)$可能有几种情况

$$

1.\ x_i=x_i^{(1)}...x_i^{(n)},p(x\vert y_i)=\prod_j p(x^{(j)}=x_i^{(j)}\vert y_i)\\

2.\ x_i=x_i^{(1)}...x_i^{(n_i)},p(x\vert y_i)=\prod_j p(x^{(j)}=x_i^{(j)}\vert y_i)

$$

第一种情况称为multivariate event model。特别的，当$p(x^{(j)}=x_i^{(j)}\vert y_i)$符合bernoulli分布时称为multivariate bernoulli event model，符合multinomial分布时称为multivariate multinomial event model

第二种情况称为multinomial event model

计算分布的公式(假设是离散的)

$$

p(x_j\vert y_i)=\frac{num\ of\ x_j\ in\ y_i}{num\ of\ y_i}\\

p(y_i)=\frac{num\ of\ y_i}{\sum_j num\ of\ y_j}

$$



为了避免0概率的影响，使用laplace平滑

$$

p(x=1)=\frac{num(x=1)+1}{(num(x=1)+1)+(num(x=0)+1)}

$$
