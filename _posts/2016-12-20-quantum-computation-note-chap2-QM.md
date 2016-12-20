# 量子力学基础

量子力学虽然是一个不够完整的理论（至少在测量问题上我还没发现能让我完全信服的解释...），但它还是经过了几十年实验的检验。从理论的角度看，它很优美，从实用的角度看，它确实可以为我们提供一些经典世界中没有的东西，所以我个人觉得量子力学还是很值得学习的。

与通常给大众的印象不同，量子力学的基本原理其实并不困难，或者说其数学直观并不困难。科学家口中量子力学的困难，更可能来自于其违背宏观世界直觉的表现，或者其在解释实际系统时计算的困难。下面主要介绍量子力学的两种表述方式，一是量子态(物理对象的状态)为Hilbert空间中矢量的形式，二是量子态为密度矩阵的形式，前者更为直观，后者有更好的计算上的性质。

# 作为Hilbert空间中矢量的量子态

## 对量子力学的直觉认识

首先要说明一点，很多人印象中量子力学充满了随机性。但实际上，除了测量问题以外，量子力学的数学过程都是确定性的（实际上测量问题也有一些不那么“随机”的解释），我们完全可以用没有随机性的方式来看待封闭系统中的量子态。

在我们的感官世界中，物质有一些在某一时刻总是取确定“**实数**”的“**值**”的属性，如位置、动量等，但在量子力学中，我们认为，这些“实数”仅仅概括了这些属性的一些性质（主要是统计上的性质），但并不足以完全表达这些属性，这些属性实际上需要用“**复数**”的“**向量**”来表示。举两个例子。

首先是一个物理上不够严谨的例子。抛开量子力学，在经典物理中，我们在宏观的层面有“温度”这一概念，我们可以给空间的每一点都定义一个属性——温度。但实际上，我们知道温度来自于微观粒子的运动，空间中某点**温度的数值** 是该点附近所有粒子运动信息（**动量、能量的数值**）的**统计概括**。量子力学中，我们认为**粒子的位置**是其在空间中某种属性（**波函数的数值**）的**统计概括**。

再者，在线性代数中，我们常常会提到一个矩阵的“秩”或者“迹”，或是向量的模长，显然，这些属性并不能完全地表示一个矩阵，我们眼中向量的**模长**与**向量**本身的关系，就类似于量子力学中一个粒子的**位置**与其**波函数**的关系。

然而，往下学习可能会有接下来的疑问。对于温度的概念，我们知道一个粒子的能量波动几乎不会造成宏观上该点附近温度数值的变化，但是量子力学中我们可以恰当地构造一个系统，使得宏观系统的状态对其中的**某个**微观粒子的状态非常敏感（类似人为设计的蝴蝶效应），薛定谔的猫就是这样一个思想实验。既然微观粒子的表述都比我们想象中的复杂，宏观世界的复杂度直觉上会指数级地增加。为什么宏观世界中一个“**实数**”就能这么好地描述物体的位置，为什么我们不能同时存在于寝室和图书馆，为什么我们没见过一只同时既是死又是活的猫？这个问题我还没学清楚。。。不过可以参考知乎上的讨论（看完后面以后再看）

https://www.zhihu.com/question/21565372

## 量子力学的基本假设

下面介绍量子力学中的“公理”。

## 假设 1: 关于物理对象的数学表述

任何孤立的物理对象（物理系统），其所有的可能状态，一一对应一个复数上带内积的线性空间（**Hilbert空间**）中的**单位向量**。我们称这个空间为该物理对象的状态空间(state space)，称这个系统被这个空间中相应的状态矢量(state vector)描述。

注意，这一假设只给出了物理对象状态空间的性质(Hilbert空间)，但没有告诉我们任何一个特定的物理对象其具体的状态空间是什么。实际上，寻找实际的物理系统对应的状态空间在数学上通常是困难的，不过好在假想的简单状态也足够我们学习量子力学的基础了。

举一个直观的例子（不严谨），我们研究的物理对象是一个被约束在一条直线$x$轴上的孤立的粒子，其状态空间的基底是其所有可以处在的位置$\vert x\rangle$，前面乘复数的线性组合$\sum_x c_x\vert x\rangle,\sum_x\vert \vert c_x\vert \vert ^2=1$。注意，这里不管$x$是多少，$\vert x\rangle$都表示一个单位向量，Dirac符号中的$x$并不参与运算，$x$不同表示这个单位向量属于不同的"维度"，“维度”相同的向量内积为$1$，不同为$0$（构成标准正交基）。也就是说，$\vert 0\rangle$不同于$\frac{1}{\sqrt{2}} (\vert -1\rangle+\vert +1\rangle)$，并且它们还是正交的。

不难看出，这样的表述中，粒子的一个状态

$$

\vert \psi\rangle=\sum_x c(x)\vert x\rangle

$$

正好对应一个实数域上的复值函数(函数可看作无限维向量)，$c:\mathbb{R}\to \mathbb{C}$，并且

$$

1=\langle \psi\vert \psi\rangle=\sum_x c(x)c(x)^*=\int_{-\infty}^{+\infty}\vert \vert c(x)\vert \vert ^2dx

$$

对应的

$$

\vert x\rangle = \delta(x)

$$

在这里$\delta$表示Dirac函数(冲激函数)。通常，我们称$c(x)$为该粒子的波函数。对于波函数，有这样的物理意义：对粒子的位置进行测量，测得$x$的几率正比于$\vert \vert c(x)\vert \vert ^2$。（“测量”这一概念我们会在下文说明）

这个例子里，状态空间是无限维的，也就是说，经典力学中只需要一个**实数**表示的一维位置，量子力学中我们需要用一整个**实数域上的复值函数**来表示。出于简单考虑，接下来我们一般研究有限维的状态空间，如一个电子在$z$轴方向的自旋，只有两维。我们可以这样表示这个状态空间：考虑一个被约束得只有$z$轴方向自旋这一自由度的电子，假设当电子$z$轴方向的自旋朝正向时其状态为$\vert 0\rangle$，负向时状态为$\vert 1\rangle$，则电子所有的可能状态为

$$

a\vert 0\rangle+b\vert 1\rangle\\ for\ a,b\in \mathbb{C}\ and\ \vert a\vert^2+\vert b\vert^2=1

$$

我们可以这样理解

$$

\vert 0\rangle=(1,0)^T\\
\vert 1\rangle=(0,1)^T

$$

这样，状态空间就是复数上的模长为$1$的二维向量空间了

## 假设2: 关于物理对象状态随时间变化的方式

给定孤立的物理系统，对于其在不同时刻$t_1$和$t_2$所处的两个量子态$\vert \psi_1\rangle$和$\vert \psi_2\rangle$，存在一个只依赖于$t_1$和$t_2$的unitary矩阵$U$使得

$$

\vert \psi_2\rangle=U\vert \psi_1\rangle

$$

类似于假设1，量子力学没有直接给出对于一个特定(particular)的物理系统，随时间演化的矩阵$U$是怎样的，只规定了$U$的形式一定是unitary的。

上面我们给出的是矩阵形式的时间演化，这一演化还有一个更有名的表示方式——薛定谔方程。

$$

i\hbar \frac{d\vert\psi\rangle}{dt}=H\vert \psi\rangle

$$

其中$H$是仅和该封闭体系有关的固定Hermitian矩阵。

这两种表示方式的对应本质上是Hermitian和unitary矩阵的一一对应，下面我们来说明这种对应。

对于Hermitian矩阵

$$

H=\sum_E E\vert E\rangle\langle E\vert

$$

考虑unitary矩阵

$$

\begin{aligned}
U(t_1,t_2)&\equiv exp[\frac{-iH(t_2-t_1)}{\hbar}]\\
&=\sum_E exp[\frac{-iE(t_2-t_1)}{\hbar}]\vert E\rangle\langle E\vert
\end{aligned}

$$

和解

$$

\vert \psi_2\rangle=U(t_1,t_2)\vert \psi_1\rangle

$$

不难验证，上式是薛定谔方程的解，这是Hermitian矩阵到unitary矩阵的对应。

考虑任意unitary矩阵$U$，我们知道$U$的特征值一定都是$exp[i\theta]$的形式($\theta$都是实数)，这样我们可以把$U$分解成

$$

U=\sum exp[i\theta]\vert \theta\rangle\langle \theta\vert

$$

的形式，那么我们有

$$

H=log(-iU)=\sum \theta\vert \theta\rangle\langle\theta\vert

$$

是一个Hermitian矩阵(可以对角分解并且特征值都是实数的矩阵)。这是unitary矩阵到Hermitian矩阵的对应。

以上介绍了孤立系统的时间演化，可以看到到目前为止，量子力学还是很有道理(reasonable)的。

接下来要介绍一种不是孤立系统中的情况，“测量”，这是我目前感觉最奇怪的地方。

## 假设3: 关于测量

一个量子测量是特征空间上线性变换的一个集合$\{M_m\}$，满足完备性条件：

$$

\sum_m M_m^\dagger M_m=I

$$

下标$m$用来编号经过测量后系统的状态，如果一个量子系统初态是$\vert \psi\rangle$，经过测量后会变为$\{\vert \psi_m\rangle\}$中的元素之一，其中

$$

\vert \psi_m\rangle=\frac{M_m\vert\psi\rangle}{\sqrt{\langle\psi\vert M_m^\dagger M_m\vert\psi\rangle}}

$$

变为$\vert \psi_m\rangle$的概率为

$$

p(m)=\langle\psi\vert M_m^\dagger M_m\vert\psi\rangle

$$

不难看出，完备性条件其实是对$\sum_m p(m)=1$的约束。

这个对测量的定义与通常量子力学教材中的投影测量(projective measurement)有一些区别，不难看出它把unitary变换也包括进来了，不过数学上这个测量的定义等价于unitary变换加上projective measurement。

据我所知，物理上还没有对测量的精确定义，只有类似这样的模糊定义：**测量是宏观的经典物体与量子体系发生的作用**。

我们再介绍一下投影测量。投影测量指的是一个Hermitian矩阵$M$，它的本征值分解是

$$

M=\sum_m mP_m

$$

其中$P_m$是到本征空间$m$的投影算符，$\{m\}$是测量可能得到的值的集合。测量后量子态以$p(m)=\langle\psi\vert P_m\vert\psi\rangle$的概率变为$\vert \psi_m\rangle=\frac{P_m\vert\psi\rangle}{p(m)}$，通常认为我们可以从与封闭体系作用的的宏观体系的改变中“读”出测量后的$m$。不难看出，假设3中定义的测量满足$M_m M_n=\delta_{mn}M_m$，即测量算符两两正交时，就能得到投影测量。

投影测量会导致量子态变换到互相正交的多个量子态之一，我们知道互相正交的量子态有一个很好的性质：我们可以有效地区分它们。关于典型的投影测量，可以了解一下**Stern-Gerlach实验**(很容易看懂，大致是理论上应该连续分布的轨道分成了明显可区分的两条)。

Stern-Gerlach实验说的就是假设1中提到的电子$z$轴方向的自旋体系，对应的测量矩阵即是

$$

Z=\vert 0\rangle\langle 0\vert - \vert 1\rangle\langle 1\vert=
\begin{bmatrix}
1 & 0\\
0 & -1\\
\end{bmatrix}

$$

假设3的测量到投影测量的对应可以这样看出来，对任意矩阵$M_m$，由矩阵的polar decomposition总有

$$

M_m=U_m\sqrt{M_m^\dagger M_m}=U_m\sum_j m_jP_j

$$

物理中我们把投影测量称为可观测量的测量(其对应矩阵的本征值是实数)，容易得出$m$的均值

$$

E(m)=\sum_m mp(m)=\sum_m m\langle\psi\vert P_m\vert\psi\rangle=\langle\psi\vert M\vert\psi\rangle

$$

方差

$$

\Delta(m)^2=E((m-E(m))^2)=E(m^2)-E(m)^2=\langle\psi\vert M^2\vert\psi\rangle-(\langle\psi\vert M\vert\psi\rangle)^2

$$

接下来我们可以介绍著名的不确定性原理。

## 不确定性原理

对Hermitian矩阵$A,B$以及$[A,B]=AB-BA,\{A,B\}=AB+BA$，假设对某个$\vert \psi\rangle$有$\langle\psi\vert AB\vert\psi\rangle=x+iy$，我们有$\langle\psi\vert BA\vert\psi\rangle=(\langle\psi\vert AB\vert\psi\rangle)^\dagger =x-iy$，那么

$$

\vert\langle\psi\vert [A,B]\vert\psi\rangle\vert^2+\vert\langle\psi\vert \{A,B\}\vert\psi\rangle\vert^2=4\vert\langle\psi\vert AB\vert\psi\rangle\vert^2

$$

由柯西不等式有

$$

\vert\langle\psi\vert A\ B\vert\psi\rangle\vert^2\leq \langle\psi\vert A^2\vert\psi\rangle \langle\psi\vert B^2\vert\psi\rangle

$$

联合以上两式有

$$

\vert\langle\psi\vert [A,B]\vert\psi\rangle\vert^2 \leq 4\langle\psi\vert A^2\vert\psi\rangle \langle\psi\vert B^2\vert\psi\rangle

$$

由于$A,B$的任意性，我们令$A=C-\langle\psi\vert C\vert\psi\rangle I,B=D-\langle\psi\vert D\vert\psi\rangle I$，可以得到

$$

\Delta(C)\Delta(D)\geq \vert\langle\psi\vert [C,D]\vert\psi\rangle\vert/2

$$

我们常听到的版本是这样描述的：$C$是位置测量，$D$是动量测量，在某些假设下我们可以求出$[C,D]\neq 0$，从而得到位置和动量无法同时精确测量的结论。然而这样的说法会造成一些误解，似乎是对位置和动量两者之一的测量影响了另一者的测量。其实一种更为准确的看法是，假设我们能准备很多个精确处于相同状态$\vert \psi\rangle$的粒子（虽然这一点并不一定能做到），我们对其中的一部分粒子进行$C$测量，对其他的粒子进行$D$测量，则测量的统计结果应该满足上式。这体现了某种更为本质的随机性。

封闭系统中的量子态总是以unitary变换来演化，这意味着这样的时间演化是确定的、可逆的，与经典力学类似。然而在测量中，量子力学突然有了随机、不可逆的现象，这带来了某种不一致性。实际上，仔细想想肯定会有这样的疑问：怎样的尺度才叫经典物体？多强的作用才足以引发测量的效应？把宏观的经典系统和微观的量子系统看作一个孤立系统，这个系统的演化是不是unitary的？

当然，这也与我们目前的实验能力有关，目前只有unitary+测量的体系可以精确地预言我们的实验结果(统计上)，我们还没有探索宏观与微观模糊地带的实验。总之，测量问题是一个大坑，现在还没有(据我所知)得到广泛承认的解释。

可以看看https://www.zhihu.com/question/21565372

## 假设4: 关于复合系统的量子态

假设系统A的状态空间是$V_A$，系统B的状态空间是$V_B$，则两个系统的复合系统的状态空间为$V_A\otimes V_B$。更具体的，假设我们知道系统1,2,...,n分别处于状态$\vert \psi_1\rangle,\vert \psi_2\rangle,...,\vert \psi_1\rangle$，则它们整体的状态是$\vert \psi_1\rangle\otimes\vert \psi_2\rangle\otimes...\otimes\vert \psi_1\rangle$。

由于张量积的性质，多粒子的体系比起单粒子的体系有很多奇特的性质，比如无法写成两个单粒子态张量积的Bell态。

$$

\frac{\vert 00\rangle+\vert 11\rangle}{\sqrt{2}}

$$

以上就是我们对量子力学第一种数学形式的介绍的全部了。在这种形式中，我们认为任一封闭体系中，物理实在(如一个或多个粒子)在某一时刻的状态(如动量、“位置”、能级、自旋等等)总是可以用唯一的态矢量表示的，并且他们的状态总是随着时间按unitary(没有与经典体系发生作用时)或者测量的方式变化。

