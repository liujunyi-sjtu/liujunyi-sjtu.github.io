---
layout: post
---

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
> $$
> \begin{bmatrix}
> cos\frac{\gamma}{2} & -sin\frac{\gamma}{2}\\
> sin\frac{\gamma}{2} & cos\frac{\gamma}{2}\\
> \end{bmatrix}
> $$
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
> $$
> \vert\psi\rangle\otimes\vert s\rangle
> $$
> 克隆的过程可以表述如下：
> $$
> \vert\psi\rangle\otimes\vert s\rangle\xrightarrow{U}U(\vert\psi\rangle\otimes\vert s\rangle)=\vert\psi\rangle\otimes\vert \psi\rangle
> $$
> 再假设我们可以用$U$克隆另一个量子态$\vert\varphi\rangle$，我们可以得到：
> $$
> U(\vert\psi\rangle\otimes\vert s\rangle)=\vert\psi\rangle\otimes\vert \psi\rangle \\
> U(\vert\varphi\rangle\otimes\vert s\rangle)=\vert\varphi\rangle\otimes\vert\varphi\rangle
> $$
> 取上面两式的内积可以得到：
> $$
> \langle\psi\vert\varphi\rangle=(\langle\psi\vert\varphi\rangle)^2
> $$
> 解得：
> $$
> \langle\psi\vert\varphi\rangle=0\ or\ 1
> $$
> 即$\vert \psi\rangle=\vert \varphi\rangle$或者$\vert \psi\rangle$与$\vert \varphi\rangle$正交，亦即$U$最多只能克隆特定的一个量子态或者其正交。再加上我们事先不知道输入的态$\vert \psi\rangle$，故可得到结论：能克隆任意量子态的unitary matrix $U$是不存在的。

