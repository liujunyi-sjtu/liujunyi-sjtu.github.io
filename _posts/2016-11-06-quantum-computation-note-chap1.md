---
layout: post
---

Michael A. Nielsen, Isaac L. Chuang-Quantum Computation and Quantum Information_ 10th Anniversary Edition-Cambridge University Press (2011)的读书笔记

# 导论

# 历史

### 物理

- 不可克隆理论
- 单量子系统的完全控制

### 计算机

- Church-Turing thesis的变迁
  - 概率图灵机形式
  - 量子计算形式
  - 。。。
- Shor和Grover的算法
- 传统计算机模拟量子力学系统的困难

## 未来的方向

- 从物理的角度思考计算
- 从计算的角度思考物理：研究复杂的系统的性质

# 量子比特qubit

量子计算中将qubit抽象为数学对象(向量)。单个qubit状态为$\vert 0\rangle$和$\vert 1\rangle$的线性叠加

$$

\vert \psi\rangle=\alpha\vert 0\rangle+\beta\vert 1\rangle
=
\begin{bmatrix}
\alpha\\
\beta\\
\end{bmatrix}

$$

状态空间可以视作二维单位复向量空间，其中$\vert 0\rangle$和$\vert 1\rangle$为两个正交基。相对于普通比特，我们无法测试一个量子比特所处的状态$\alpha$和$\beta$，而只能以正比于系数平方的概率测到相应的本征态($\vert \alpha\vert^2$概率测到$\vert 0\rangle$，$\vert \beta\vert^2$概率测到$\vert 1\rangle$，normalization: $\vert\alpha\vert^2+\vert\beta\vert^2=1$)。

- qubit的state无法直接观测
- 我们对qubit的测量结果受到其state的影响

**定义**


$$

\vert +\rangle=\frac{1}{\sqrt{2}}\vert 0\rangle+\frac{1}{\sqrt{2}}\vert 1\rangle\\
\vert -\rangle=\frac{1}{\sqrt{2}}\vert 0\rangle-\frac{1}{\sqrt{2}}\vert 1\rangle

$$


### qubit状态空间的一种几何图像(Bloch Sphere)

由于$\vert\alpha\vert^2+\vert\beta\vert^2=1$，qubit的状态可以如下表述

$$

\vert \psi\rangle=e^{i\gamma}(cos\frac{\theta}{2}\vert 0\rangle+e^{i\varphi}sin\frac{\theta}{2}\vert 1\rangle)

$$

由于项$e^{i\gamma}$没有可观测效应，我们可以认为

$$

\vert \psi\rangle=cos\frac{\theta}{2}\vert 0\rangle+e^{i\varphi}sin\frac{\theta}{2}\vert 1\rangle

$$


即三维单位球面上的一点($z$轴正向表示$\vert 0\rangle$负向表示$\vert 1\rangle$)。

## 多量子比特

两个qubits的状态

$$

\vert \psi\rangle=\alpha_{00}\vert 00\rangle+\alpha_{01}\vert 01\rangle+\alpha_{10}\vert 10\rangle+\alpha_{11}\vert 11\rangle

$$

可以只测量一个粒子，如果测量结果为0，则状态变为

$$

\vert \psi'\rangle=\frac{\alpha_{00}\vert 00\rangle+\alpha_{01}\vert 01\rangle}{\sqrt{\vert\alpha_{00}\vert^2+\vert\alpha_{01}\vert^2}}

$$

**定义：Bell state (or EPR pair)**

$$

\frac{\vert 00\rangle+\vert 11\rangle}{\sqrt{2}}

$$

此状态中，两个粒子的状态是相关联的，如果第一个粒子测得$0$则随后对第二个粒子也一定测得$1$，对$1$亦然。（纠缠态）

对有n个量子比特的系统，共需要$2^n$个参数来描述。

# 量子计算

描述量子状态变化的语言。

经典计算机可以用逻辑门和连接它们的线路表示，量子计算机也可以抽象为量子门和线路。

- 线路：传递信息
- 门：操作信息

## Single qubit gates

不同于经典bit只有一种单元素门(NOT)，单个qubit有无数种非平凡的门。

### NOT门


$$

X=
\begin{bmatrix}
0 & 1\\
1 & 0\\
\end{bmatrix}\\
X\begin{bmatrix}\alpha\\ \beta\\ \end{bmatrix}=\begin{bmatrix}\beta\\ \alpha\\ \end{bmatrix}

$$


NOT门的线性性来自于量子力学的特性。

由于normalization($\vert\alpha'\vert^2+\vert\beta'\vert^2=1$)的需求，对单量子操作的量子门对应的矩阵U应该满足

$$

(U\begin{bmatrix}\alpha\\ \beta\\ \end{bmatrix})^\dagger U\begin{bmatrix}\alpha\\ \beta\\ \end{bmatrix}=1

$$

其中$\dagger$表示取复共轭和转置。由此可以解得

$$

U^\dagger U=I

$$

即U是酉unitary矩阵，这是对量子门的唯一限制。这就是说对任意一个酉矩阵都有相对应的亮子们。

### Z门


$$

Z=
\begin{bmatrix}
1 & 0\\
0 & -1\\
\end{bmatrix}\\

$$


### Hadamard门


$$

H=
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 1\\
1 & -1\\
\end{bmatrix}\\

$$


性质

$$

H\vert 0\rangle=\frac{1}{\sqrt{2}}\vert 0\rangle + \frac{1}{\sqrt{2}}\vert 1\rangle \\
H\vert 1\rangle=\frac{1}{\sqrt{2}}\vert 0\rangle - \frac{1}{\sqrt{2}}\vert 1\rangle

$$

几何上相当于Bloch Sphere先绕$y$轴转$90^\circ$再绕$x$轴转$180^\circ$.可以用来构造叠加态(?)

任何一个$2\times2$unitary矩阵都可以如下分解

$$

U=e^{i\alpha} 
\begin{bmatrix}
e^{-i\beta/2} & 0\\
0 & e^{i\beta/2}\\
\end{bmatrix}
\begin{bmatrix}
cos\frac{\gamma}{2} & -sin\frac{\gamma}{2}\\
sin\frac{\gamma}{2} & cos\frac{\gamma}{2}\\
\end{bmatrix}
\begin{bmatrix}
e^{-i\delta/2} & 0\\
0 & e^{i\delta/2}\\
\end{bmatrix}\\

$$

并且我们不需要对所有的$\alpha,\beta,\gamma$都构造相应的量子门，而只需要一些特殊的$\alpha,\beta,\gamma$对应的门，就可以用它们来逼近任意的量子门。

> $$
> \begin{bmatrix}
> e^{-i\beta/2} & 0\\
> 0 & e^{i\beta/2}\\
> \end{bmatrix}
> $$
>
> 可以看作一个相位偏移(phase shift)
>
> $$
> \begin{bmatrix}
> cos\frac{\gamma}{2} & -sin\frac{\gamma}{2}\\
> sin\frac{\gamma}{2} & cos\frac{\gamma}{2}\\
> \end{bmatrix}
> $$
>
> 是二维平面上的一个旋转。(Bloch Sphere上的几何意义不明确(?))

进一步的，对任意数量的量子比特进行的任意计算，都可以由属于一个有限集合的量子门生成。这样的集合称为量子计算的一个universal set。

## Multiple qubit gates

### CNOT门(controlled-NOT)

多qubits量子门的原型(prototype)

$$

U_{CN}=
\begin{bmatrix}
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 0 & 1\\
0 & 0 & 1 & 0
\end{bmatrix}

$$

作用于：

$$

(\alpha_{00},\alpha_{01},\alpha_{10},\alpha_{11})

$$

其中第一个qubit称为control qubit，第二个成称为target qubit。当第一个qubit设为0时，第二个qubit不做变化；第一个qubit设为1时，第二个qubit取反，即：

$$

\vert 00\rangle\to\vert 00\rangle;\vert 01\rangle\to\vert 01\rangle;\vert 10\rangle\to\vert 11\rangle;\vert 11\rangle\to\vert 10\rangle

$$

CNOT也可以理解成XOR的一般化：

$$

\vert A,B\rangle\to\vert A,B\oplus A\rangle

$$

经典门：AND, OR, XOR, NAND, NOR。经典的门在qubit中没有直接的对应，因为它们是不可逆的（信息丢失）。量子门对应的必须是unitary matrix，即量子门总是可逆的。在量子计算中，我们需要用可逆的方式来进行经典的逻辑。

### Universality

经典情况：任何经典逻辑门都可以由NAND组成。

量子情况：任何multiple qubit logic gate都可以由CNOT和single qubit gates组成。

## 量子电路quantum circuits

量子电路中的wire并不一定对应着物理的线路，也可能是时间的推移，或者粒子从一处移动到另一处。

与经典电路的区别：

- 不允许‘loops’，即反馈。量子电路是无环的
- 不允许线路交叉(多个线路交叉得到一条线路)
- 上一条操作的逆也不允许(一条线路分叉得到多条)

## qubit的复制问题

对传统比特，利用CNOT可以实现复制：

$$

(x,0)\xrightarrow{CNOT}(x,x\oplus0)=(x,x)

$$

对于量子比特$\vert \psi\rangle=\alpha\vert 0\rangle+\beta\vert 1\rangle$，尝试给CNOT类似的如下输入：

$$

\vert \psi\rangle\vert 0\rangle=\alpha\vert 00\rangle+\beta\vert 10\rangle

$$

将会得到：

$$

\vert \psi\rangle\vert 0\rangle\xrightarrow{CNOT}\alpha\vert 00\rangle+\beta\vert 11\rangle

$$

而复制的目标是：

$$

\vert \psi\rangle\vert \psi\rangle=\alpha^2\vert 00\rangle+\alpha\beta\vert 01\rangle+\alpha\beta\vert 10\rangle+\beta^2\vert 11\rangle

$$

可见对传统比特的复制方法在此失效了。

这样不可复制的性质是量子信息和经典信息的主要差异之一。

另一个看待复制失败的角度是，当我们对$\alpha\vert 00\rangle+\beta\vert 11\rangle$中任一qubit进行测量时，两个qubit都会坍缩到$0$或者$1$，这样我们关于$\alpha$和$\beta$的隐藏信息就丢失了。而如果复制成功，这些信息应该得以保留。

> ### 不可克隆原理的初等证明(对unitary操作而言)
>
> 记我们要克隆的量子态为$\vert\psi\rangle$，另一个同时输入的需要将其转变为$\vert\psi\rangle$的态为$\vert s\rangle$(相当于克隆的输出对象)，记我们的克隆机器为unitary matrix $U$。
>
> 机器的输入为：
>
> $$
> \vert\psi\rangle\otimes\vert s\rangle
> $$
>
> 克隆的过程可以表述如下：
>
> $$
> \vert\psi\rangle\otimes\vert s\rangle\xrightarrow{U}U(\vert\psi\rangle\otimes\vert s\rangle)=\vert\psi\rangle\otimes\vert \psi\rangle
> $$
>
> 再假设我们可以用$U$克隆另一个量子态$\vert\varphi\rangle$，我们可以得到：
>
> $$
> U(\vert\psi\rangle\otimes\vert s\rangle)=\vert\psi\rangle\otimes\vert \psi\rangle \\
> U(\vert\varphi\rangle\otimes\vert s\rangle)=\vert\varphi\rangle\otimes\vert\varphi\rangle
> $$
>
> 取上面两式的内积可以得到：
>
> $$
> \langle\psi\vert\varphi\rangle=(\langle\psi\vert\varphi\rangle)^2
> $$
>
> 解得：
>
> $$
> \langle\psi\vert\varphi\rangle=0\ or\ 1
> $$
>
> 即$\vert \psi\rangle=\vert \varphi\rangle$或者$\vert \psi\rangle$与$\vert \varphi\rangle$正交，亦即$U$最多只能克隆特定的一个量子态或者其正交。再加上我们事先不知道输入的态$\vert \psi\rangle$，故可得到结论：能克隆任意量子态的unitary matrix $U$是不存在的。


## Bell states

指以下线路生成则状态$\vert\beta_{xy}\rangle$(纠缠态)：

$$

\vert H(x)\rangle\vert y\rangle\xrightarrow{CNOT}\vert\beta_{xy}\rangle \\
\vert\beta_{xy}\rangle=\frac{\vert 0,y\rangle+(-1)^x\vert 1,\overline{y}\rangle}{\sqrt{2}}

$$


## 量子传输 quantum teleportation

有一个待传输的qubit$\vert \psi\rangle$和一个Bell state如$\vert \beta_{00}\rangle$构成的系统$\vert \psi_0\rangle=\vert \psi\rangle\vert \beta_{00}\rangle$，可以通过量子电路将$\vert \psi\rangle$转移到Bell state的一个qubit上得到$\vert x\rangle\vert y,\psi\rangle$。

# 量子算法

## 量子线路模拟经典线路

### Toffoli gate

是一个可逆的经典门

$$

(a,b,c)\xrightarrow{T}(a,b,c\oplus ab)

$$

可以用它来模拟NAND和FANOUT(复制一个信号到两路)从而构成所有经典电路

$$

NAND:(a,b,1)\xrightarrow{T}(a,b,\neg(ab))\\
FANOUT:(1,a,0)\xrightarrow{T}(1,a,a)\\

$$

由于Toffoli gate可逆，因此存在一个unitary matrix U恰好实现了Toffoli gate，即存在这样的量子门。

由此，量子电路应该可以模拟经典的逻辑电路(但是好像没有寄存器，怎么实现迭代和递归(?))

## 量子并行

二元函数$f(x):\{0,1\}\to\{0,1\}$的实现，其中x称为data register，y称为target register

$$

U_f:\vert x,y\rangle\to\vert x,y\oplus f(x)\rangle

$$

在后面的章节中会证明，$U_f$是可以有效得出的。考虑如下过程

$$

\frac{\vert 0\rangle+\vert 1\rangle}{\sqrt{2}}\vert 0\rangle\xrightarrow{U_f}\frac{\vert 0,f(0)\rangle+\vert 1,f(1)\rangle}{\sqrt{2}}

$$

这一过程同时计算了$f$在不同输入上的输出。然而当我们对结果的状态进行测量时，我们只能得到一个输出。

### Walsh-Hadamard transform

即对多个输入的qubit进行$H$操作后，作为n元函数的输入。通过这一变换，我们可以将上述一元函数的情况扩展到任意元函数。以$n=2$为例：

$$

(\frac{\vert 0\rangle+\vert 1\rangle}{\sqrt{2}})(\frac{\vert 0\rangle+\vert 1\rangle}{\sqrt{2}})=(\frac{\vert 00\rangle+\vert 01\rangle+\vert 10\rangle+\vert 11\rangle}{2})

$$

我们用$H^{\otimes 2}$表示以上操作。更一般的，Hadamard变换作用于n个$\vert 0\rangle$的结果为

$$

\frac{1}{\sqrt{2^n}}\sum_x \vert x\rangle

$$

其中$\sum$包括了x的所有取值($2^n$个)，这一操作记为$H^{\otimes n}$。这样，上述的函数操作可以扩展到对n个qubits输入的形式。对n+1个状态为$\vert 0\rangle^{\otimes n}\vert 0\rangle$的qubit，对前n个进行Hadamard变换并应用$U_f$后，产生的state为

$$

\frac{1}{\sqrt{2^n}}\sum_x \vert x\rangle\vert f(x)\rangle

$$

同样，当我们对结果进行测量时，也只能得到一个输出。这样的输出在经典计算机上也能有效地得到，因此对量子并行这一现象的利用还需要进一步的方法。

## Deutsch's algorithm

对给定的函数$f$考虑如下过程：

$$

\left.\begin{aligned}
\vert 0\rangle\xrightarrow{H}\\
\vert 1\rangle\xrightarrow{H}\\
\end{aligned}\right\}
\vert \psi_1\rangle
\xrightarrow{U_f}
\vert \psi_2\rangle
\left\{\begin{aligned}
\xrightarrow{H}\\
\xrightarrow{\ }\\
\end{aligned}\right\}
\vert \psi_3\rangle

$$

即：

$$

\vert \psi_1\rangle=(H\vert 0\rangle)(H\vert 1\rangle)\\
\vert \psi_2\rangle=U_f\vert \psi_1\rangle=\vert \psi_{20}\rangle\vert \psi_{21}\rangle\\
\vert \psi_3\rangle=(H\vert \psi_{20}\rangle)\vert \psi_{21}\rangle

$$





第一步：

$$

\vert \psi_1\rangle=(\frac{\vert 0\rangle+\vert 1\rangle}{\sqrt{2}})(\frac{\vert 0\rangle-\vert 1\rangle}{\sqrt{2}})

$$

第二步，枚举$\vert \psi_2\rangle$：

| f(1)\f(0) | 0          | 1           |
| --------- | ---------- | ----------- |
| 0         | (0+1)(0-1) | -(0-1)(0-1) |
| 1         | (0-1)(0-1) | -(0+1)(0-1) |

可以总结得到

$$

\vert \psi_2\rangle=
\left\{\begin{aligned}
\pm (\frac{\vert 0\rangle+\vert 1\rangle}{\sqrt{2}})(\frac{\vert 0\rangle-\vert 1\rangle}{\sqrt{2}})&,f(0)=f(1)\\
\pm (\frac{\vert 0\rangle-\vert 1\rangle}{\sqrt{2}})(\frac{\vert 0\rangle-\vert 1\rangle}{\sqrt{2}})&,f(0)\neq f(1)\\
\end{aligned}\right.

$$

第三步，作用H门于第一个qubit得到：

$$

\vert \psi_3\rangle=
\left\{\begin{aligned}
\pm \vert 0\rangle(\frac{\vert 0\rangle-\vert 1\rangle}{\sqrt{2}})&,f(0)=f(1)\\
\pm \vert 1\rangle(\frac{\vert 0\rangle-\vert 1\rangle}{\sqrt{2}})&,f(0)\neq f(1)\\
\end{aligned}\right.

$$

即

$$

\vert \psi_3\rangle=\pm \vert f(0)\oplus f(1)\rangle(\frac{\vert 0\rangle-\vert 1\rangle}{\sqrt{2}})

$$

在这里，我们只通过一次$U_f$的计算就得到了关于$f$的global的性质$f(0)\oplus f(1)$，看起来这里我们得到了一个比经典计算机更好的结果。

## Deutsch-Jozsa algorithm

对Deutsch问题在n个qubit输入上的扩展。

假设$f:\{0,1\}^n\to\{0,1\}$，并且$f$(x)的取值只可能是(1) constant或者(2)0和1各一半(balanced)。假设我们有$U_f$可以计算$\vert x,y\rangle\to\vert x,y\oplus f(x)\rangle$，那么我们可以用下列算法只调用一次对$f$的计算$U_f$确定$f$是上述情况中的哪一种。

$$

\left.\begin{aligned}
\vert 0\rangle^{\otimes n}&\xrightarrow{H^{\otimes n}}\\
\vert 1\rangle&\xrightarrow{H}\\
\end{aligned}\right\}
\vert \psi_1\rangle
\xrightarrow{U_f}
\vert \psi_2\rangle
\left\{\begin{aligned}
\xrightarrow{H^{\otimes n}}\\
\xrightarrow{\ }\\
\end{aligned}\right\}
\vert \psi_3\rangle

$$

计算过程如下：

$$

\vert \psi_1\rangle=\sum_x\frac{\vert x\rangle}{\sqrt{2^n}}[\frac{\vert 0\rangle-\vert 1\rangle}{\sqrt{2}}]

$$

又有$\vert x\rangle[\frac{\vert 0\rangle-\vert 1\rangle}{\sqrt{2}}]\xrightarrow{U_f}(-1)^{f(x)}\vert x\rangle[\frac{\vert 0\rangle-\vert 1\rangle}{\sqrt{2}}]$，得到：

$$

\vert \psi_2\rangle=\sum_x\frac{(-1)^{f(x)}\vert x\rangle}{\sqrt{2^n}}[\frac{\vert 0\rangle-\vert 1\rangle}{\sqrt{2}}]

$$

我们可以这样表示Hadamard操作：$H\vert x\rangle=\sum_z(-1)^{xz}\vert z\rangle/\sqrt{2}$，由此

$$

\vert \psi_3\rangle=\sum_z\sum_x\frac{(-1)^{x\cdot z+f(x)}\vert z\rangle}{\sqrt{2^n}}[\frac{\vert 0\rangle-\vert 1\rangle}{\sqrt{2}}]

$$

注意到state $\vert 0\rangle^{\otimes n}$的系数是$\sum_x(-1)^{f(x)}/2^n$，当$f(x)$是常数时，得到$\vert 0\rangle^{\otimes n}$的系数是$1$或者$-1$，由normalization的条件，我们可以得知其他项的系数都是$0$，亦即我们对前$n$个qubits只能测到$\vert 0\rangle^{\otimes n}$。而当$f(x)$是balanced时，$\vert 0\rangle^{\otimes n}$的系数则正好是$0$，亦即前$n$个qubits不可能测得$\vert 0\rangle^{\otimes n}$，或者说至少有一位是$1$。

由此，我们对$\vert \psi_3\rangle$的测量可以把$f$的两种情况分开来，原问题通过对$f$的一次计算就可以得到解答。

如果使用经典计算机运行确定性算法，此问题最坏情况下需要对$f$进行$2^{n-1}+1$次（当然此处的计算和$U_f$有很大的不同，直接比较有些不妥）。但是使用经典计算机运行概率性算法，此问题也能在很快的时间内得到不错的解答（参考Miller-Rabin素数测试）。

由此，我们得到了第一个在量子计算机上表现优于经典计算机的问题。

# 量子算法总结

目前优于经典计算机的量子算法主要有三类

- 基于量子版本的傅里叶变换：D-J算法和Shor算法
- 基于量子搜索算法
- 量子模拟：模拟量子系统

## 基于傅里叶变换

离散傅里叶变换(DFT)定义：$x_0,...,x_{N-1}\to y_0,...,y_{N-1}$

$$

y_k=\frac{1}{\sqrt{N}}\sum^{N-1}_{j=0}e^{2\pi ijk/N}x_j\\
x_j=\frac{1}{\sqrt{N}}\sum^{N-1}_{k=0}e^{-2\pi ijk/N}y_k

$$

在量子计算中，$n$个$qubits$的状态空间有$2^n$个互相正交的基(量子力学中不同的本征态总是正交的)，我们使用以下形式的DFT：

$$

\vert j\rangle \to \frac{1}{\sqrt{2^n}}\sum^{2^n-1}_{k=0}e^{-2\pi ijk/2^n}\vert k\rangle

$$

相当于对$2^n$维向量的变换。此时我们有如下性质得到保持：

$$

\sum^{2^n-1}_{j=0}x_j\vert j\rangle\to \frac{1}{\sqrt{2^n}}\sum^{2^n-1}_{k=0}[\sum^{2^n-1}_{j=0}e^{-2\pi ijk/2^n}x_j]\vert k\rangle=\sum^{2^n-1}_{k=0}y_k\vert k\rangle

$$

在经典计算机中，我们进行$N=2^n$维的DFT，需要$Nlog(N)=n2^n$次运算(FFT)，而量子计算的傅里叶变换只需要$\log^2(N)=n^2$次运算，这比起经典计算机得到了指数级的加速。

然而需要注意的是，量子计算中傅里叶变换的结果仍是叠加态，这意味着我们并不能得到每一项的系数$y_k$，而只能将它们作为隐藏信息使用。这是这一方法难以利用的原因。

这一类算法可以解决这样的问题：

给定$f:G\to X$，其中$G$是一个有限生成群，X是一个有限集合，对$G$的子集$K$，$f$在K的每一个陪集上取常数，并且不同的陪集对应的常数不同。假设存在$U_f$使得$U_f\vert g\rangle\vert h\rangle=\vert g\rangle\vert h\oplus f(g)\rangle$，其中$g\in G,h\in X$，$\oplus$是$X$上恰当选定的二元运算，在此前提下找一个$K$的生成集。

Deutsch-Jozsa算法和Shor算法都可以看作以上问题的特例。

## 量子搜索算法

首先由Grover发现。对于一个大小为N的搜索空间，在对其结构没有先验知识的情况下，在其中搜索一个有满足属性的元素，经典的计算机需要大约N次的操作。而量子搜索算法则能在$\sqrt{N}$次操作中得到结果。

## 量子模拟

对于量子系统的模拟，量子计算机可以用$kn$个qubits来模拟经典计算机需要$c^n$个bits来模拟的量子系统。但相对的，我们可能无法直接从量子计算机模拟的结果中得到我们想要的信息(在叠加态的隐藏信息中)。这需要我们恰当地设计。

## 量子计算的能力

一些现状：

- 目前认为质因数分解可能不属于P，但也不属于NPC
- 基于搜索的量子算法不能有效地解决所有NP问题，但不排除NP问题的某些深层结构可以让量子计算机快速解决它们。
- BQP问题是量子计算中被认为可以有效解决的一类问题(类似经典计算中的P问题)。目前并不明确BQP和P，NP，PSPACE的关系，但已知$P\subseteq BQP\subseteq PSPACE$(等号能否取到还未知)。
