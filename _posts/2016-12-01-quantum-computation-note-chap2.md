---
layout: post
---


# 线性代数

量子力学中通常用$\vert v\rangle$这样的符号来表示列向量(实质上是线性空间中的一个元素)，用$\langle v\vert$来表示其共轭转置得到的行向量$((\vert v\rangle)^*)^T$

比如，如果

$$

\vert v\rangle=
\begin{bmatrix}
v_1\\
\vdots\\
v_n
\end{bmatrix}

$$

那么

$$

\langle v\vert=
\begin{bmatrix}
v_1^*&\cdots &v_n^*
\end{bmatrix}

$$


## 线性空间

...

## 内积

定义：$(\cdot,\cdot):V\times V\to R$

1. 线性性：

   $(\vert v\rangle,\sum_i \lambda_i\vert w_i\rangle)$=$\sum_i \lambda_i(\vert v\rangle,\vert w_i\rangle)$

2. $(\vert v\rangle,\vert w\rangle)=(\vert w\rangle,\vert v\rangle)$

3. $(\vert v\rangle,\vert v\rangle)\geq 0$当且仅当$\vert v\rangle=0$的时候成立

简写：$\langle v\vert w\rangle\equiv (\vert v\rangle, \vert w\rangle)$

定义了内积的线性空间称为内积空间(也称为Hilbert space)。

## 线性变换

我们在讨论线性变换之前，要先定好讨论的范围，即线性空间。通常我们讨论的是有限维的线性空间，即存在一组有限个线性无关的向量，该空间中任意向量都可以表示为这组向量的线性组合，我们称这组向量为一个基底。

线性变换的定义，对$A:V\to W$，如果

$$

A(\sum a_i\vert v_i\rangle)=\sum a_iA(\vert v_i\rangle)

$$

则我们称$A$为$V$到$W$的一个线性变换

对一组正交基底，我们可以发现$A$的表现正好类似一个矩阵，此时我们可以用矩阵来表示$A$

假设

$$

A\vert v_j\rangle=\sum_i A_{ij}\vert w_i\rangle

$$

可以得到

$$

A_{ij}=\langle w_i\vert A\vert v_j\rangle\\
A=\sum_{ij} A_{ij}\vert w_i\rangle\langle v_j\vert

$$


## Pauli矩阵


$$

\sigma_0=
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
\end{bmatrix}\\
\sigma_1=\sigma_X=
\begin{bmatrix}
0 & 1 \\
1 & 0 \\
\end{bmatrix}\\
\sigma_2=\sigma_Y=
\begin{bmatrix}
0 & -i \\
i & 0 \\
\end{bmatrix}\\
\sigma_3=\sigma_Z=
\begin{bmatrix}
1 & 0 \\
0 & -1 \\
\end{bmatrix}\\

$$




##  Gram–Schmidt方法

可以从任意一组基$\vert w_i\rangle$得到一组该线性空间上的标准正交基$\vert v_i\rangle$

$$

\vert v_1\rangle=\vert w_1\rangle\\

\vert v_{k+1}\rangle=\frac{\vert w_{k+1}\rangle-\sum_{i=1}^k\langle v_i\vert w_{k+1}\rangle\vert v_i\rangle}{\lVert\vert w_{k+1}\rangle-\sum_{i=1}^k\langle v_i\vert w_{k+1}\rangle\vert v_i\rangle\rVert}

$$


## 外积Outer Product

定义线性变换$\vert w\rangle\langle v\vert$为

$$

(\vert w\rangle\langle v\vert) (\vert v'\rangle) \equiv \vert w\rangle\langle v\vert v'\rangle=\langle v\vert v'\rangle\vert w\rangle

$$

这个变换相当于一个投影，我们称之为外积

对一组完备正交基$\vert i\rangle$我们有

$$

\sum_i\vert i\rangle\langle i\vert =I

$$

对线性变换$A:V\to W$

$$

A=I_W AI_V=\sum_{ij}\langle w_j\vert A\vert v_i\rangle\vert w_j\rangle\langle v_i\vert

$$

投影可以反应线性空间中的很多性质。

## 特征值和特征向量(本征值和本征向量，Eigenvectors and eigenvalues)

在给定的线性空间中，对线性变换$A$，如果对非零向量$\vert v\rangle$有$A\vert v\rangle=v\vert v\rangle$，则称$\vert v\rangle$为$A$的特征值为$v$的特征向量。我们一般通过以下方式求解$A$的特征向量：

对本征向量，我们有$(A-\lambda I)\vert \lambda\rangle=0$，这意味着方程有非零解，意即$\vert A-\lambda I\vert = 0$，这是一个关于$\lambda$的多项式，由复数域上多项式至少有一个根我们得知$A$必然存在至少一个本征值和相应的 本征向量。

我们称本征值为$v$的向量构成的空间为$v$对应的本征空间(eigenspace)，一个本征空间可能多于一维，此时我们称之为退化的。

对一些矩阵（正规矩阵），我们可以将之分解为本征向量上的投影之和$A=\sum_i\lambda_i\vert i\rangle\langle i\vert$，其中$\vert i\rangle$为$\lambda_i$对应的正交对角化过的本征向量(不同的$i$可能有相同的$\lambda_i$)，我们称这样的矩阵是可对角化的(diagonalizable)。

## 伴随和厄米变换Adjoints and Hermitian operators

对内积空间上的线性变换$A$，总存在一个变换$A^\dagger$使得对线性空间中的任意两个向量有：

$$

(\vert v\rangle,A\vert w\rangle)=(A^\dagger\vert w\rangle,\vert v\rangle)

$$

我们称$A^\dagger$为$A$的伴随或者厄米共轭。

$A^\dagger$的存在性可以这样得出

$$

\begin{aligned}
(\vert v\rangle,A\vert w\rangle) & =(\sum v_i^*\langle i\vert) A(\sum w_j\vert j\rangle)\\
&=(\sum v_i^*\langle i\vert) (\sum_j w_j\sum_k A_{kj}\vert k\rangle)\\
&=(\sum v_i^*\langle i\vert) (\sum_k \vert k\rangle\sum_j A_{kj}w_j)\\
&=\sum A_{ij}v_i^*w_j\\

(A^\dagger\vert v\rangle,\vert w\rangle) & =(\vert w\rangle,A^\dagger\vert v\rangle) ^*\\
&=(\sum A^\dagger_{ij}w_i^*v_j)^*\\
&=\sum (A^\dagger_{ji})^*v_i^*w_j
\end{aligned}

$$

比较系数我们可以得知$(A^\dagger_{ji})^*=A_{ij}$，即$A^\dagger=(A^*)^T$

一个伴随是自身的变换我们称之为厄米变换或自伴变换。投影projector是一类重要的自伴变换，定义如下

假设$k$维线性空间$W$是$d$维线性空间$V$的一个子空间，我们总可以用Gram–Schmidt方法找到$W$上的一组标准正交基，$\vert 1\rangle,...,\vert k\rangle$，并在$V$上构造包含这$k$的基底的一组$d$个基底$\vert 1\rangle,...,\vert d\rangle$，这时我们称

$$

P\equiv \sum_{i=1}^k\vert i\rangle\langle i\vert

$$

为一个到$W$上的投影，显然$P$是厄米的。我们称

$$

Q\equiv I-P

$$

为$P$的正交补。

定义矩阵$U$是unitary的，当$U^\dagger U=I$。对unitary变换$U=\sum_i \vert e_i\rangle\langle v_i\vert$，线性变换$A$，$B=UAU^\dagger$，容易验证，$B$在$\vert e_i\rangle$上和$A$在$\vert v_i\rangle$上表现一致，可以认为$A$和$B$表示的是同一种线性变换，只是对应的基底不同，因此我们称这种变换为unitary similarity transformation。

### 线性变换在不同基底上的矩阵表示

对线性变换$A$，它在基底$\vert v_i\rangle$下的矩阵表示为$A'$，在基底$\vert w_j\rangle$下的矩阵表示为$A''$，我们有

$$

\begin{aligned}
A&=A\sum_j\vert v_j\rangle\langle v_j\vert=\sum_j\sum_i A'_{ij}\vert v_i\rangle\langle v_j\vert\\
&=\sum_l\sum_k A_{kl}''\vert w_k\rangle\langle w_l\vert
\end{aligned}

$$

我们得到

$$

A_{ij}'=\sum_l\sum_k A_{kl}''\langle v_i\vert w_k\rangle\langle w_l\vert v_j\rangle

$$


## 正规变换

定义：我们称$A$为正规变换，如果

$$

AA^\dagger=A^\dagger A

$$

正规变换有很好的性质：$A$是正规变换当且仅当$A$是可对角化的。

我们通过归纳$A$的维数来证明。

假设$P$是到$A$的特征值为$\lambda$的特征空间的投影，$Q\equiv I-P$，我们有

$$

\begin{aligned}
AP&=\lambda P\\
A&=(P+Q)A(P+Q)=PAP+QAQ+PAQ+QAP\\
QAP&=\lambda QP=0\\
\end{aligned}

$$

由于$AA^\dagger=A^\dagger A$，我们有

$$

AA^\dagger P=A^\dagger AP=\lambda A^\dagger P

$$

从而$A^\dagger P$ 也只在特征空间$P$中，我们有

$$

PAQ=(QA^\dagger P)^*=(\lambda QP)^*=0

$$

因此对正规变换A有

$$

A=PAP+QAQ

$$

而$PAP=\lambda PP=\lambda P$已经是对角化的了，我们只需要说明$QAQ$是对角化的。又由于$QAQ$维数小于$A$，我们只需要证明$QAQ$也是正规的就可以利用归纳假设来证明。利用前面的结论

$$

QA=QA(P+Q)=QAQ\\
QA^\dagger=QA^\dagger(P+Q)=QA^\dagger Q

$$

我们有

$$

QAQQA^\dagger Q=QAA^\dagger Q=QA^\dagger AQ=QA^\dagger QQAQ

$$

即$QAQ$正规。$\Box$

## 半正定变换

我们称对任意$\vert v\rangle$都有$\langle v\vert A\vert v\rangle\geq0$的线性变换$A$为半正定变换。

半正定变换一定是厄米的，从而是正规的，验证如下：

首先，任何矩阵$A$都可以表示成如下

$$

A=\frac{A+A^\dagger}{2}+i\frac{A-A^\dagger}{2i}=B+iC

$$

容易验证，B和C都是厄米的。

对$\vert v\rangle$，我们有

$$

\langle v\vert A\vert v\rangle=\langle v\vert B\vert v\rangle+i\langle v\vert C\vert v\rangle

$$

由于$\vert v\rangle$任意，而$\langle v\vert A\vert v\rangle$是实数，我们有对任意$\vert v\rangle$

$$

\langle v\vert C\vert v\rangle=0

$$

从而$C=0$，$A=B$，即A厄米。

## 正规变换，厄米变换，半正定变换的一些性质

- 半正定变换=>厄米变换=>正规变换
- 正规变换都可以被对角化
- 一个正规变换是厄米的，当且仅当它的特征值都是实数
- 一个正规变换是半正定的，当且仅当它的特征值都非负
- 属于不同特征值的特征向量彼此线性无关
- 厄米变换的属于不同特征值的特征向量彼此正交
- 任意形式为$A^\dagger A$的线性变换都是半正定的

## 张量积Tensor product

假设$V$和$W$分别为维数$m$和$n$的线性空间，为了方便我们假设$V$和$W$都是Hilbert空间，那么定义$V\otimes W$是一个$mn$维的线性空间，其中的元素是$\vert v\rangle\otimes\vert w\rangle$的线性组合。

譬如说$\vert v_i\rangle$是$V$的一组基，$\vert w_j\rangle$是$W$的一组基，我们有$\vert v_i\rangle\otimes\vert w_j\rangle$是$V\otimes W$的一组基。

$\vert v\rangle\otimes\vert w\rangle$经常简写为$\vert v\rangle\vert w\rangle,\vert v,w\rangle,\vert vw\rangle$。

张量积的性质：

1. $z(\vert v\rangle\otimes\vert w\rangle)=(z\vert v\rangle)\otimes\vert w\rangle=\vert v\rangle\otimes (z\vert w\rangle)$，这体现了张量积不同于笛卡尔积。
2. $(\vert v_1\rangle+\vert v_2\rangle)\otimes\vert w\rangle=\vert v_1\rangle\otimes\vert w\rangle+\vert v_2\rangle\otimes\vert w\rangle$
3. $\vert v\rangle\otimes(\vert w_1\rangle+\vert w_2\rangle)=\vert v\rangle\otimes\vert w_1\rangle+\vert v\rangle\otimes\vert w_2\rangle$

我们可以定义$V\otimes W$上的线性变换。首先我们定义

$$

(A\otimes B)(\vert v\rangle\otimes\vert w\rangle)=(A\vert v\rangle)\otimes(B\vert w\rangle)

$$

容易验证，$A\otimes B$是线性变换。

对任意线性变换$C:V\otimes W\to V'\otimes W'$，我们都可以将其表示成

$$

C=\sum_i A_i\otimes B_i

$$

其中$A_i:V\to V',B_i:W\to W'$。

我们可以这样构造：令$A^{(ij)}=\vert v’_j\rangle\langle v_i\vert,B^{(kl)}=\vert w’_l\rangle\langle w_k\vert$，容易验证$A^{(ij)}\otimes B^{(kl)}$的线性组合可以表示任何的$C$。

定义$V\otimes W$上的内积如下

$$

(\vert v\rangle\otimes \vert w\rangle,\vert v'\rangle\otimes \vert w'\rangle)=\langle v\vert v'\rangle\langle w\vert w'\rangle

$$

在给定基底下，张量积有很直观的矩阵表示。对$m\times n$矩阵$A$和$p\times q$矩阵$B$

$$

A\otimes B=
\begin{bmatrix}
A_{11}B&\cdots &A_{1n}B\\
&\vdots&\\
A_{m1}B&\cdots &A_{mn}B\\
\end{bmatrix}

$$


## 线性变换的函数

给定函数$f:C\to C$，对正规变换$A=\sum_i \lambda_i\vert i\rangle\langle i\vert$，定义

$$

f(A)=\sum_i f(\lambda_i)\vert i\rangle\langle i\vert

$$

例子：

$$

exp(\theta \sigma_Z)=
\begin{bmatrix}
e^\theta & 0\\
0 & e^{-\theta}\\
\end{bmatrix}

$$


## 迹trace

定义$A$的迹为其对角元素之和($\vert i\rangle$为一组标准正交基)

$$

tr(A)\equiv \sum_i A_{ii}=\sum_i \langle i\vert A\vert i\rangle

$$

容易验证$tr(AB)=tr(BA)$，

$$

\begin{aligned}
tr(AB)&=\sum_i \langle i\vert AB\vert i\rangle\\
&=\sum_i \langle i\vert A(\sum_j \vert j\rangle\langle j\vert )B\vert i\rangle\\
&=\sum_{ij}\langle i\vert A\vert j\rangle\langle j\vert B\vert i\rangle\\
&=\sum_{ji}\langle j\vert B\vert i\rangle\langle i\vert A\vert j\rangle\\
&=tr(BA)
\end{aligned}

$$

由此，对unitary矩阵$U$有$tr(UAU^\dagger)=tr(U^\dagger UA)=tr(A)$，可见迹在unitary similarity transformation下是一个不变量。

对任意标准正交基$\vert i\rangle$，我们有

$$

tr(A\vert \psi\rangle\langle \psi\vert)=\sum_i \langle i\vert A\vert \psi\rangle \langle \psi\vert i\rangle=\langle \psi\vert A\vert \psi\rangle

$$

这一结果十分实用。

定义$L_V$为Hilbert空间$V$上的线性变换的集合，我们可以定义线性变换的内积

$$

(A,B)\equiv tr(A^\dagger B)

$$

称为Hilbert-Schmidt inner product

## 对易子和反对易子commutator and anti-commutator

$A,B$的对易子定义为

$$

[A,B]=AB-BA

$$

如果$[A,B]=0$，我们称$A$和$B$是对易的。

反对易子定义为

$$

\{A,B\}=AB+BA

$$

对于对易的厄米变换，有如下的重要性质：

对厄米变换$A$和$B$，$[A,B]=0$当且仅当存在一组标准正交基$\vert i\rangle$使得$A=\sum_i a_i\vert i\rangle\langle i\vert$，$B=\sum_i b_i\vert i\rangle\langle i\vert$

证明如下

对矩阵$A$的特征值为$a$的特征空间$V_a$的一组标准正交向量$\vert a,j\rangle$($j$用来表示特征空间退化时不同特征向量的编号)，我们有

$$

AB\vert a,j\rangle=BA\vert a,j\rangle=aB\vert a,j\rangle

$$

可以发现经过$B$变换后，该特征向量依然在这个特征空间中。设$P_a$是到$V_A$上的投影，容易验证$B_a\equiv P_a BP_a$是一个厄米变换，从而$B_a$可以在$V_a$中对角化为

$$

B_a=\sum_b b\sum\vert a,b,k\rangle\langle a,b,k\vert

$$

其中$\vert a,b,k\rangle$表示$B_a$的特征值为$b$的特征空间的一组标准正交基。容易验证$\vert a,b,k\rangle$也是$V_a$中的一组向量。这样，我们得到了$V_a$上$A,B$共同特征向量组成的一组标准正交基。

我们对$A$所有的特征值$a$都寻找这样的一组标准正交基$\vert a,b,k\rangle$，由于

$$

\sum_a B_a=(\sum_a P_a)B(\sum_a P_a)=IBI=B

$$

我们也就得到了一个$B$的对角化分解。此时对所有的$\vert a,b,k\rangle$构成了$A$和$B$可以共同对角化的基底。

### Pauli矩阵的对易规律

$\epsilon_{jkl}$表示排列Levi-Civita符号

$$

[\sigma_j,\sigma_k]=2i\sum_{l=1}^3 \epsilon_{jkl}\sigma_l\\

\{\sigma_j,\sigma_k\}=2\delta_{jk}I\\

\sigma_j\sigma_k=\delta_{jk}I+i\sum_{l=1}^3 \epsilon_{jkl}\sigma_l\\

$$


## 极分解和奇异值分解The polar and singular value decomposition

一个一般的线性变换的性质通常并不直观，但利用下面的结果，我们可以把任何一个线性变换分解为一个unitary变换和一个半正定变换的乘积，这两者的性质非常的直观。

### 极分解

对$V$上的任意线性变换$A$，存在unitary变换$U$和半正定变换$J,K$使得

$$

A=UJ=KU

$$

其中$J\equiv \sqrt{A^\dagger A},K\equiv \sqrt{AA^\dagger}$。并且如果A可逆，那么$U$是唯一的

我们称$UJ$为$A$的左极分解，$KU$为A的右极分解。

证明如下：

由于$A^\dagger A$半正定，其总可以被分解成$A^\dagger A=\sum \lambda_i^2 \vert i\rangle\langle i\vert$的形式，其中$\vert i\rangle$是一组标准正交基。我们令$J\equiv \sum \lambda_i \vert i\rangle\langle i\vert$，对$\lambda_i\neq0$令$\vert e_i\rangle=A\vert i\rangle / \lambda_i$。我们有

$$

\begin{aligned}
\langle e_i\vert e_i\rangle&=\langle i\vert A^\dagger A\vert i\rangle/\lambda_i^2=1\\
for\ i\neq j, \langle e_j\vert e_i\rangle&=\langle j\vert A^\dagger A\vert i\rangle/\lambda_i\lambda_j=0\\
\end{aligned}

$$

因此$\vert e_i\rangle$是标准正交的。利用 Gram–Schmidt方法，我们可以将$\vert e_i\rangle$扩充成一组$V$上的标准正交基。我们令

$$

U\equiv \sum_i\vert e_i\rangle\langle i\vert

$$

容易验证$U$是unitary的。考虑任意$\vert i\rangle$，我们有

$$

\begin{aligned}
for\ \lambda_i\neq 0,UJ\vert i\rangle&=U(J\vert i\rangle)=U(\lambda_i\vert i\rangle)=\lambda_i\vert e_i\rangle=A\vert i\rangle\\
for\ \lambda_i=0,UJ\vert i\rangle&=0=A\vert i\rangle\\
\end{aligned}\\

$$

注意$for\ \lambda_i=0,\langle i\vert A^\dagger A\vert i\rangle=0\Rightarrow\lVert(A\vert i\rangle)\rVert=0\Rightarrow A\vert i\rangle=0$

由于$\vert i\rangle$是一组基底，我们可以得到左极分解

$$

A=UJ

$$



对于给定的$A$，$J$是唯一的，这是由于$A^\dagger A=J^\dagger U^\dagger UJ=J^2$，由此$J=\sqrt{A^\dagger A}$。容易看出，当$A$可逆时，$J$也是可逆的，此时有$U=AJ^{-1}$，也就是说，当$A$可逆时，$U$是唯一的。

由于$U$是unitary的，我们有

$$

A=UJ=UJU^\dagger U=(UJU^\dagger)U

$$

令$K\equiv UJU^\dagger$，我们得到$A$的右极分解

$$

A=KU

$$


### 奇异值分解

关于极分解的讨论并没有涉及到线性变换的矩阵表示。在给定标准正交基底下，线性变换$A$通过极分解得到的矩阵还可以进一步分解成更简单的矩阵形式。

当我们涉及到矩阵表示时，存在对角矩阵这么一类简单的矩阵。

对给定的基底$\vert v_i\rangle$，对角矩阵指的是如下形式的矩阵

$$

\begin{bmatrix}
d_{11}&0&\cdots&0\\
0&d_{22}&\cdots&0\\
\vdots\\
0&0&0&d_{nn}
\end{bmatrix}

$$

即只有对角线上可以存在非零数的矩阵，其对应的线性变换呈如下的形式

$$

D=\sum_i d_{ii}\vert v_i\rangle\langle v_i\vert

$$

下面我们讨论矩阵的奇异值分解。

任意方阵$A$都可以分解为如下形式：

$$

A=UDV

$$

其中$U,V$为unitary矩阵，$D$为对角线上元素都非负的对角矩阵。我们称$D$对角线上的非零元素为$A$的奇异值。

考虑基底为$\vert v_i\rangle$时，矩阵$A$对应的线性变换的左极分解

$$

A=SJ=(\sum_i\vert e_i\rangle\langle i\vert)(\sum \lambda_i \vert i\rangle\langle i\vert)

$$

考虑unitary矩阵

$$

T=\sum_i \vert i\rangle\langle v_i\vert

$$

和对角矩阵

$$

D=\sum_i \lambda_i\vert v_i\rangle\langle v_i\vert

$$

容易验证，有

$$

J=T D T^\dagger

$$

从而

$$

A=SJ=STDT^\dagger=(ST)D T^\dagger

$$

容易验证，$ST$是unitary的。因此我们令

$$

\begin{aligned}
U &= ST\\
V &= T^\dagger
\end{aligned}

$$

即可得到$A$的奇异值分解

$$

A=UDV

$$

