---
layout: post
date:   2016-11-05 21:00:00 +0800
---

交大CS课程 可计算理论 的复习笔记。由于是复习时所做，笔记较为粗糙，待有时间再维护。

参考材料：

1. Yuxi Fu所作课程讲义
2. Cutland, Nigel. Computability: An introduction to recursive function theory. Cambridge university press, 1980

# lec00 introduction
几个目前不可能的问题
- 一个程序是否有循环
- 解决所有数论问题的证明器
- 一台可以计算现有计算机不能计算的问题（可计算性上的不可能）的计算机
- 

# lec01 可计算函数
## algorithm
algorithm包括了一套有限的指令，给定定义域内的一个输入，算法可以在有限的时间内得出一套相应的输出。

## URM模型
- 有无限个寄存器$R_1...R_n$对应的值为$r_1...r_n$，除了包含输入的前n个以外均初始化为0
- 有一个对应的程序program，包含有限条指令，每个指令是以下之一

instruction | response
---|---
Z(n) | $r_n:=0$
S(n) | $r_n:=r_n+1$
T(m,n) | $r_n:=r_m$
J(m,n,q) | if $r_n=r_m$ goto the q-th instruction; else goto the next

### configuation
指的是目前所有寄存器的值和当前执行到的指令编号

- $P(a_1,a_2,...)$以指定初始状态运行程序
- $P(a_1,a_2,...)\downarrow b$ P在指定初始状态运行收敛到b，即停机时$r_0=b$
- $P(a_1,a_2,...)\uparrow$P在制定初始状态运行发散
- $P(a_1,a_2,...,a_m)=P(a_1,a_2,...,a_m,0,0,...)$
- 
P计算了函数f指的是

$$

P(a_1,a_2,...,a_m)\downarrow b\Leftrightarrow f(a_1,a_2,...,a_m)=b

P(a_1,a_2,...,a_m)\uparrow\Leftrightarrow (a_1,a_2,...,a_m)\notin Dom(f)
$$

一个函数有无穷个程序来实现（加入废指令）


### n元predicate
相当于一个谓词，数学上是一个$\mathbb{N}^n$的子集M，当一个n元组属于这个集合M时，我们说predicate M在这个n元组上是真的。

### 可判定predicate 
predicate M可判定当且仅当特征函数

$$

c_M(x)=\left\{
\begin{array}{lcl}
1, &M(x)\ holds\\
0, &otherwise
\end{array}
\right.
$$

可计算

f: D->D可计算当且仅当存在可计算a: D->N和a^-1:N->D和可计算f'使得

$f'=a\circ f\circ a^{-1}$

# lec2 生成可计算函数
基本函数

$$

zero:0(x)=0

successor:S(x)=x+1

projectionU^n_i(x_1,...,x_n)=x_i

$$

程序的标准型
对长度为s的程序P，标准型为其中所有J(m,n,q)的q<=s+1

P[l1,...->l]将l_1到l_n放到第1_n个寄存器，其他清零，再运行P，P的计算结果放到第l个寄存器

substitution 参数代换

recursion

$$

h(x,0)=f(x)
h(x,y+1)=g(x,y,h(x,y))
$$

由基本函数开始，经过substitution,recursion组合得到的函数都可计算。

没有用到$\mu$都是原始递归的，ackermann函数不是原始递归的

# lec3 Church's Thesis
所有直观的对计算模型的定义，都会得到和URM模型一样的可计算函数集合

# lec4 哥德尔编码
- X denumerable: 存在双射f:X->N
- X enumerable: 存在满射g:N->X
- effectively denumerable: 双射f:X->N和逆映射都可计算

$$
\pi(m,n)=2^m(2n+1)-1

\pi^{-1}(x)=(\pi_1(x), \pi_2(x))

\pi_1=(x+1)_1

\pi_2=((x+1)/2^{\pi_1(x)}-1)/2

$$

几个e.d.
1.

$$

N\times N\times N\rightarrow N

\xi(m,n,q)=\pi(\pi(m-1, n-1), q-1)

\xi^{-1}(x)=\pi(\pi_1(\pi_1(x))+1, \pi_2(\pi_1(x))+1,\pi_2(x)+1)
$$

2.

$$
\bigcup_{k>0}\mathbb{N}^k

$$

## 哥德尔编码
所有的指令都可以有效编码到N(通过取4的模)

所有有限的指令序列，都可以有效编码到N

故每一个程序P都能唯一对应到一个自然数$\gamma(P)$


- $\phi_n$: 由编码为n的程序计算的函数
- $W_n$: 上述函数定义域
- $E_n$: 上述函数值域

如果$f=\phi_a$，称a是f的一个index，一个函数有无数个index

### 对角线方法：
存在一个不可计算的total function

### s-m-n theorem
对可计算函数$\phi_e^{m+n}(x,y)$存在一个函数$s_n^m$使得$\phi_e^{m+n}(x,y)=\phi_{s_n^m(e,x)}^{n}(y)$

给定了e,可以构造先对前m个寄存器初始化，然后再计算P_e的程序，并且可以有效地算出这个程序的编码。

$$

T(n,m+n)

...

T(1,m+1)

Q(1,x_1)

...

Q(m,x_m)

P_e

Q(i,x)=Z(i),S(i),...,S(i)\ for\ x\ of S(i).
$$

# 通用程序
$\phi_U(e,x_1,...,x_n)=\phi_e(x_1,...,x_n)$可计算

以下三个函数可计算，并且是primitive recursive
- c_n(e,x,t)第e个程序运行t步后的寄存器编码
- j_n(e,x,t)第e个程序运行t步后的下一个指令序号，停止了则为0
- $\sigma_n=\pi(c_n,j_n)$

下列命题可判定
- $S_n(e,x,y,t)=(j_n(e,x,y)=0\ and\ (c_n(e,x,t))_1=y$: P_e(x)在t步或更少步收敛于y
- H_n(e,x,t): P_e(x)在t步或更少步收敛
- 

## Kleene's Normal Form
$T_n(e,x,z)=S_n(e,x,(z)_1,(z)_2)$

$\phi_e^{(n)}(x)=(\mu zT_n(e,x,z))_1$

每个可计算函数都可以通过对一个primitive recursive使用最多一次$\mu$得到

## 应用
$\phi_x\ is\ total$不可判定，用对角线方法

存在一个不是primitive recursive的函数，通过对primitive recursive的函数有效编码，将编码映射到哥德尔编码，再用对角线构造可得。

所有机器可计算的函数都是递归函数，所有递归函数都是可计算的。

# lec6 可判定，不可判定，半可判定
## 可判定
特征函数可计算

不可判定即非可判定

$x\in W_x$不可判定，通过反证法，用对角线方法构造矛盾

$x\in W_e$和$x\in E_e$都不可判定，

$$

h(x)=\left\{
\begin{array}{ll}
x, &x\in W_x\\
undefined, &x\notin W_x
\end{array}
\right.
$$

x in Dom(h) iff x in W_x iff x in Ran(h)

由此，停机问题不可解

不可判定的问题
- $\phi_x=0$
- $\phi_x=\phi_y$
- $c\in W_x$
- $c\in E_x$
- 

### Rice's Therom
$\phi_x\in B$其中B是任一非空非全的可计算函数集合，这一命题不可判定。

## 半可判定

$$

f(x)=\left\{
\begin{array}{lcl}
1, &M(x)\ holds\\
undefined, &otherwise
\end{array}
\right.
$$

称M半可判定

半可判定的问题
- 停机问题
- 任何可判定的问题
- $x\in\phi_e$
- $x\in E_y$
- $W_x\neq \emptyset$

M半可判定当且仅当存在可计算函数g使得$M(x)\Leftrightarrow x\in Dom(g)$

M半可判定当且仅当存在半可判定的R使得$M(x)\Leftrightarrow \exists y R(x,y)$

一个命题M是可判定的当且仅当他的正反命题都是半可判定的

# lec7 递归集，递归可枚举集

$$

c_A(x)=\left\{
\begin{array}{lcl}
1, &x\in A\\
0, &x\notin A
\end{array}
\right.
$$

对集合A，若$c_A$可计算则称集合A是递归的 recursive

$$

f_A(x)=\left\{
\begin{array}{lcl}
1, &x\in A\\
undefined, &x\notin A
\end{array}
\right.
$$

对集合A，若$f_A$可计算则称集合A是递归可枚举的 recursive enumerable(r.e.)


递归集
- asdasd

递归可枚举集
- $K=\{x\vert x\in W_x\}$

非递归可枚举的集合
- $\overline{K}=\{x\vert x\notin W_x\}$ 否则K就递归了
- $\{x\vert \phi_x\ is\ total\}$ 对角线

集合A r.e. iff A是某个可计算函数的定义域 iff A是某个可计算函数的值域

故$W_0, W_1, W_2, ...$是对r.e.集合的一个enumeration

一个集合是r.e.的 iff 存在一个半可判定predicate R(x,y)满足$x\in A\ iff\ \exists y.R(x,y)$

A is recursive iff A and not A are r.e.

以下命题等价
1. A is r.e.
2. $A=\emptyset$或者A是一个一元total可计算函数的值域
3. A是一个可计算函数的值域

4. A是一个可计算函数的定义域
5. 存在一个半可判定predicate R(x,y)满足$x\in A\ iff\ \exists y.R(x,y)$
6. A是一个可计算函数的值域

如果A, B是r.e.集合，那么$A\cap B$和$A\cup B$都r.e.

一个无穷集合是recursive的 iff 它是某个严格单调增的total可计算函数的值域。

任意无穷的r.e.集合都有一个无穷的recursive子集

$$

A=Ran(f)

g(0)=f(0)

g(n+1)=f(\mu y(f(y)>g(n)))
$$

## Rice-Shapire Theorem
对任意形如$\{x\vert \phi_x\in A\}$的函数下标集，若该集合是r.e.的，那么$f\in A$ iff 存在有限函数$\theta\subseteq f$满足$\theta\in A$

思路：在反证假设下，构造$\phi_{s(x)}\in A\Leftrightarrow x\notin W_x$，得到一个非r.e.的集合可以规约到A，从而A不可能是r.e.的，构成矛盾。

使用方式：对r.e.的集合A

1. 常使用$f_\emptyset$来构造矛盾，如果A包含$f_\emptyset$，则A必包含所有的函数，矛盾。
2. 如果A包含的函数都是无穷函数，也构成矛盾。
3. 如果A包含的函数都是有穷函数，因为会导致所有它的超集无穷函数都在A里面，也矛盾。

## productive and creative
### productive
定义：对集合A，如果存在total可计算函数g使得

$$
\forall W_x\subseteq A\Rightarrow g(x)\in A\backslash W_x

$$

则称A是productive的

取$W_x=A$可以得到productive集合一定不是r.e.的

定理：如果A productive并且存在total可计算函数f使得

$$

x\in A\Leftrightarrow f(x)\in B

$$

则B productive

思路：对$W_x\subseteq B$找到其在f下的原像集，利用A的productive函数再用f变回来即可。

productive集合
- $\{x\vert \phi_x\neq \bold0\}$
- $\{x\vert c\notin W_x\}$
- $\{x\vert c\notin E_x\}$
- 

定理：含有$f_\emptyset$的非全的函数下标集都productive

思路：考虑到空函数属于B，构造$k(x)\in B\Leftrightarrow x\in\overline{K}$

### creative
定义：对集合A，如果A r.e.，并且$\overline{A}$ productive，则称A creative

creative集合
- $\{x\vert x\in W_x\}$
- $\{x\vert c\in W_x\}$
- $\{x\vert c\in E_x\}$
- $\{x\vert \phi_x(x)=0\}$
- $\{x\vert W_x\neq \emptyset\}$

定理：非空非全的r.e.函数下标集creative

思路：如果$f_\emptyset\in A$那么A productive就必不re，故$f_\emptyset\in\overline{A}$。

### simple
并非所有的r.e.集合creative

引理：对total可计算函数g，存在total可计算函数k使得$W_{k(x)}=W_x\cup\{g(x)\}$

定理：一个productive集合包含一个无限的r.e.子集

思路：对$W_{e_0}=\emptyset$，有$W_{e_0}\subseteq A$，$k(e_0)\in A$，$W_{k^{n+1}(e_0)}=W_{k^{n}(e_0)}\cup\{g(k^n(e_0))\}=\{g(e_0),g(k(e_0)),...,g(k^n(e_0)\}\subseteq A$，由productive的性质，$g(k^{n+1}(e_0))\notin W_{k^{n+1}(e_0)}$，从而$\{g(e_0),g(k(e_0)),...,g(k^n(e_0),...\}$是A的一个无限r.e.子集

定义：A是simple的当A同时满足
1. $A$ is r.e.
2. $\overline{A}$ is infinite
3. $\overline{A}$ contains no infinite r.e. subset

构造思路：构造一个r.e.集合，首先它的补集要是无限的，然后使得它和每一个r.e.集合都有一个交集，这样它的补集就不能包含任意一个r.e.子集，因为任意一个r.e.集合都至少有一个元素不在它的补集里。

$f(x)=\phi_x(\mu z(\phi_x(z)>2x)),A=Ran(f)$
1. $A$ is r.e.
2. $\overline{A}$ is infinite. 因为$A\cap\{0,1,...,2n\}$至多包含了n个元素$\{f(0),...,f(n-1)\}$
3. $\overline{A}$ contains no infinite r.e. subset.因为每个非空$W_x$都和A有共同元素$f(x)$

# lec8
引理：对任何可判定predicate M，可以构造一个命题$\sigma$，使得M holds iff $\sigma$为真

所有真命题构成的集合为productive，所有可证命题构成的集合为r.e.

所有可以包括自然数的形式系统，都存在一个命题，它和它的反都不能证。

# lec9 规约和度
将一个问题看作一个predicate，那么A可以规约到B可以理解为，给定A的输入，通过有效的计算可以转换为B的输入，通过B的解答可以回答A的问题。

## many-one 规约
定义$A\leq_mB$：$\exists\ total\ computable\ g\ that\ x\in A\ iff\ g(x)\in B$

如果f是单射，则特别称为one-one规约

性质
1. $\leq_m$自反传递
2. $A\leq_mB\ iff\ \overline{A}\leq_m\overline{B}$
3. $A\leq_mB$ B recursive则A recursive
4. A recursive B非空非全 A可以规约到B
5. B r.e. A规约到B A也r.e.
6. 只有全集可以规约到全集，只有空集规约到空集
7. 全集可以规约到非空集合，空集可以规约到非全集合

如果A r.e.又不recursive，那么A不能规约到A反，A反也不能规约到A

用many-one规约划分的等价类叫做m-degree

- recursive m-degree是包含了一个recursive集合的
- r.e. m-degree是包含了一个r.e.集合的
- 

1. 空集和全集各单独成一个m-degree
2. recursive m-degree包含了所有的递归集和除了空集和全集
3. 一个r.e. m-degree只包括r.e.集合
4. 最大的r.e. m-degree是$d_m(K)$

定理：
对于任意a,b两个m-degree，都存在最小上界2A并2B+1

一个r.e.集合是m-complete当所有r.e.集合m规约到他

例子
- $\{x\vert c\in W_x\}$
- $\{x\vert \phi_x\in B\}$
- $\{x\vert \phi_x(x)=0\}$

K是m-complete

### Myhill's theorem
一个集合是m-complete的当且仅当它是creative的

对simple set有0_m<a<0_m'

## 图灵规约
加入A oracle以后称为A-computable

1. 

加入oracle后依然有s-m-n和universal program

$A\leq_T B$如果$c_A$是B-computable

1. 自反传递
2. $A\leq_mB\Rightarrow A\leq_T B$
3. A和非A互相图灵规约
4. A recursive A图灵规约到所有B
5. Brecursive A图灵规约到B 那么Arecursive
6. A r.e.则A图灵规约到K

T-complete: 所有r.e.都可以规约到他

等价关系划分T-degree

recursive T-degree包含一个recursive集合

只有一个最小的degree 包括所有递归集

包括K的degree是最大的r.e. degree

任何两个degree有最小上界

1. $K^A$是A-r.e.
2. B是A-r.e. $B\leq_T K^A$
3. $A<_TK^A$
4. $A\leq_TB\Rightarrow K^A\leq_TK^B$

$a'=K^A$
1. a<a'
2. a<b then a'<b'
3. B is a-r.e. then $b\leq a'$

所有的非递归r.e.度包括一个simple set

存在r.e.集合A B $a\nleq b$并且$b\nleq a$ 称a,b incomparable

对任意r.e.度a存在b ab不可比

对任意a<b有a<c<b

对惹你a 有b,c, b<a c<a and a=b并c b\vert c

# lec10 递归定理
对任意total函数f，存在n使得$\phi_{f(n)}=\phi_n$


