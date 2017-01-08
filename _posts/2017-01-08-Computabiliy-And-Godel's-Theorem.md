# 计算机模型，递归函数，停机问题和哥德尔不完备定理

# 写这篇文章的原因

由于对高中OI神犇们的印象，上大学之前，我一直以为Computer Science应该是一门很数学的专业。但是在CS的前两年里，几乎所有的功课都是以Engineering为中心展开的（甚至是Electronic Engineering），鲜有深入的数学课。这甚至让我怀疑自己选错了专业(orz)，直到在大三下学期上了由傅老师编纂教案（至少资料上是这么写的），龙环老师授课的可计算理论。这门课程围绕可计算这一中心，从很直观的URM模型出发，深入浅出地向我们展示了上世纪计算机科学领域(个人认为乃至是数学领域)最重要的一些成果。这些由Gödel，Turing等人发展的成果不光有着重要的理论意义，更有着深刻的哲学内涵，一度让我有一种柏拉图的洞中人瞥到了一眼真实的震撼。然而据我了解，国内鲜有学校这样安排这门课程，我觉得这实在是一个遗憾。为了让更多的人能了解到这些伟大的理论，也为了向傅老师的努力表示尊敬，我把课程所学简化整理成了这篇文章。

初步完成本文后，我感觉自己还是没能把这些内容讲好。由于篇幅太长加上表达不尽人意，可能很少有人能看完这篇文章，不过如果大家对这篇文章有什么建议，请务必告诉我！

# 前言

或许每一个求知路上的人都有过这样的疑问：这个问题是否有答案？究竟什么是我们能知道的，这是一个玄妙而深刻的问题。从柏拉图到康德到黑格尔，几乎每一个对理性的世界有过深入思考的哲学家都尝试过这个问题。可惜哲学往往不能给我们答案，幸好我们还有数学。希尔伯特曾经对这个问题信心满满，至少直觉上，在一个合理的数学体系中，每一个数学问题都应该存在一个答案——一个命题要么是对的，要么是错的，这无关于我们是否已经知道这个答案。然而，先是哥德尔，后是图灵，证明了希尔伯特的信心只是我们对数学过于美好的期望。

哥德尔证明了，即使是我们最“熟悉”的自然数集合中，也能找出不存在答案的问题。用数学的语言说，对于任意一套可以描述自然数的公理系统，如皮亚诺公理，要不然这个系统是自相矛盾的，要不然我们就能找到一个关于自然数的命题，我们无法用公理通过逻辑的方式证明或者证伪它。注意，如果我们把这一命题作为新的公理，新的公理系统仍然能描述自然数，我们就又能找出另一个无法证明的命题。由于自然数对我们而言太重要了，我们只能允许这种命题的存在。

而图灵则是先提出了一个有说服力的机械计算模型图灵机，然后证明了即使是对于一些有确定答案的问题，我们也没有办法找到答案。

上述的结论有着深刻的内涵，从数学上说，它似乎让我们看到了数学的不完美，却也让我们看到了数学的可能性；从哲学上说，它似乎在说我们的认知总是有限的，而真理确实无限的。

它意味着我们在某些地方必须小心地对待我们思考的对象，即使是数学中我们也必须考虑“这个问题是否有答案”，“我们能否找到答案”。

让人惊讶的是，这些深刻的结果本质上只关系到最基本的一阶谓词逻辑和自然数的性质，这也是我们得以在这篇不长的文章中讲述它们的原因。

关于以上内容的文学叙述在很多地方都能找到，而这篇文章的目标则是给有集合论基础和计算机基础知识的人（比如CS本科生）提供这一结论更为数学的直观。

这篇文章的思路大致是这样：首先介绍一个简单的计算机模型——URM模型，作为一个直观的基础。然后介绍递归函数，著名的图灵机就与这类函数等价，这一类函数有着非常重要的意义。接着用前两部分的语言介绍停机问题和Universal程序，这是图灵最重要的成果。最后介绍哥德尔不完备定理，这是上世纪数学领域最重要的成果之一，这一部分我觉得自己讲的有点混乱，原始文献[1]和参考书[2]比本文讲得好，本文可能更适合作为这两个资料的辅助材料或者中文概括...

# URM模型

在考虑数学问题之前，我们可以尝试给数学世界和物质世界构建一个联系，意即将数学世界的问题一一对应到物质世界的某种物理的现象上。图灵机就是一个成功的尝试，作为第一个足够强大的机械的计算模型，图灵机的影响力已经远远超出了计算机科学的范围。

在我看来，图灵机确实有着非常机械的直观，是一个非常有说服力的物理模型，然而在计算机科学经历了几十年的飞速发展后，对熟悉现代计算机的人来说，用图灵机来解决问题或许就不那么直观了。幸好我们还有别的选择，URM模型和lambda演算模型就是两个较为贴近现代计算机程序设计的模型。这两个模型和图灵机一样，解决问题的能力都等价于后面会介绍的递归函数。其中，URM模型非常类似于现代计算机中的汇编语言，更为贴近物理实现，因此，我们选择URM模型作为本文的计算模型。

递归函数是数学中非常重要的一类函数，我们迄今为止找到的一切计算模型，在解决问题的能力上最多等价于递归函数。由于无法证明，在Church和Turing的建议下，我们认为递归函数能解决的问题就是“数学上我们所能知道的”。这一建议称为Church-Turing Thesis。

## URM模型的定义

一个Unlimited Register Machine包括

1. 无限个有序的、能存储任意大自然数的寄存器（内存）
2. 一个有编号的指令序列（程序），其中每个指令都是四种对寄存器操作之一

具体来说，URM模型的寄存器大概是这样从$0$开始编号的可列无限个寄存器，其中每个寄存器都能存储一个任意大小的自然数

$\vert R_0\vert R_1\vert R_2\vert R_3\vert ...$

我们用$R_n$表示第$n$个寄存器，用$r_n$表示$R_n$中存储的值，我们有$n\in \mathbb{N},r_n\in \mathbb{N}$。

尽管现代计算机不能满足URM模型对“无限”的要求，但是对特定的问题，我们通常都能用足够多的有限的资源解决。相信在与现代计算机等价的问题上，URM模型还是有足够说服力的。

URM模型的四种操作是

1. Z(n)，表示将$r_n$置为0。
2. S(n)，表示将$r_n$的值加1。
3. T(m, n)，表示用$r_m$的值覆盖$r_n$的值。
4. J(i, m, n)，表示如果$r_n==r_m$那么接下来执行第$i$条指令，否则执行下一条指令。

通常我们给出一个URM模型以后，会指定其输入参数的个数。一个有n个输入参数的URM的输入是其前n个寄存器的初始值，而第n个及以后的寄存器默认置为0。URM模型的执行从第一条指令开始，对一个有$m$条指令的URM模型，当它执行到第$k>m$条指令时停机，如果程序永远都无法停机，我们称之发散。URM模型的输出是其停机时$r_0$的值。

对于程序$P$，输入$x$，如果程序输出为$b$，我们记$P(x)\downarrow b$，如果程序发散，我们记$P(x)\uparrow$。

容易看出，每一个URM程序$P$都对应一个从自然数的子集(或者自然数有序对的子集)到自然数的函数$f$。下文中我们会发现，每个$\mathbb{N}^k\to \mathbb{N}$的函数其实都正好对应一个$\mathbb{N}\to \mathbb{N}$的函数（实质上，我们可以用质因子分解的唯一性，将有限个自然数编码到一个自然数上），因此我们大部分情况下仅考虑一元函数。

如果$\forall x\in \mathbb{N}$

1. $P(x)\downarrow b\iff f(x)=b$
2. $P(x)\uparrow \iff f(x)\text{ undefined}$

我们称$P$计算了函数$f$。

如果一个函数$f$在任意$x\in\mathbb{N}$都有定义，那么我们称$f$是total function。

对于任意一个自然数上的函数$f$，如果存在一个URM程序计算它，那么我们称$f$是可计算的。

对于任意一个关于自然数的命题（谓词）$M$，如果以下函数是可计算的(称为$M$的特征函数)，我们称这个命题为可判定命题

$$

c_M(x)=
\begin{aligned}
\left\{
 \begin{aligned}
 1&\quad&\text{if $M(x)$ is true}\\
 0&\quad&\text{if $M(x)$ is false}
 \end{aligned}
 \right.
\end{aligned}

$$

是否所有自然数上的函数都有对应的URM程序呢？

我们可以这样看出上面问题的答案是否定的。从集合的基数来看，自然数上函数的集合$\{f\vert f:\mathbb{N}\to \mathbb{N}\}$大小为$\mathbb{N}^\mathbb{\vert N\vert }$。而每个URM程序总可以表示成一个字符串，我们总可以把一个字符串看成一个自然数(比如说字符串的字符集有128个，那就是128进制的自然数)，也就是说URM程序计算的函数只有$\vert \mathbb{N}\vert$个。我们知道一个集合的幂集总是严格比它本身大，因此肯定存在URM模型不能计算的自然数上的函数，而且它们比URM模型能计算的函数多得多。

## URM模型的例子

下面给出一个URM模型的"a+b"程序，相信任何一个对汇编有一点点了解的人都能看懂。

```cos
输入 r0, r1
输出 r0 = r0 + r1
//我们用r2作为一个临时变量，r0增加的值和r2的值相同，当r2的值等于r1时，r0就增加了r2那么多了
0 J(4, 1, 2) //如果r1==r2 那么程序终止，否则增加r0和r2 
1 S(0)
2 S(2)
3 J(0, 0, 0) //无条件跳转回指令0
```

其实URM模型的T指令可以由另外三种指令表示，我们可以举一个例子

```code
//由于我们有无限个寄存器，我们总能找到一个后面不会使用的寄存器Rk，用它做临时寄存器来实现T(m, n)
0 Z(n)
1 J(5, m, k)
2 S(n)
3 S(k)
4 J(1, 0, 0)
...
```

相信任何一个熟悉汇编语言的人，在上面的说明之后，都有能力用URM模型实现程序设计中常用的运算操作和控制操作（加减乘除，分支，循环等等），对URM模型与现代计算机解决问题能力的等价性（不考虑时间复杂度）有一个直观的感觉。

我们对URM模型的介绍主要是为了给下面的递归函数引入一个物质的基础，另外给递归函数的编号提供一些方便，因此对URM模型的介绍就到此为止。

# 递归函数

递归函数是一类在自然数上的非常重要的函数，我们认为递归函数就是所有我们能计算的函数，这一断言从20世纪初递归函数理论出现至今还没找到反例。

递归函数是由大名鼎鼎的Gödel，Kleene等人发展起来的，在1931年哥德尔不完备定理的论文中就有完整的定义了，或许实际出现的时间更早（我没有考证过...），个人认为图灵机的机械模型也是因为其与递归函数等价才有足够的说服力。

## 递归函数的定义

我们给出递归函数的归纳定义。先是三类基本的函数

1. $Z(n)=0$，常零函数(zero function)是递归函数。
2. $S(n)=n+1$，加一函数(successor function)是递归函数。
3. $U_k^n(x_1,...,x_k,...,x_n)=x_k$，投影函数(projection function)是递归函数，即在输入之中选择一个输出。

然后是三种构造方法

4. 函数的复合(composition)：如果$f$和$g_k,1\leq k\leq n$都是递归函数，那么$h(x)=f(g_1(x),...,g_n(x))$也是递归函数。

5. 函数的原始递归(primitive recursion)：如果$g$和$f$都是递归函数，那么定义$h$
   
$$

   \begin{aligned}
   h(x,0)&=f(x)\\
   h(x,y+1)&=g(x, y, h(x,y))
   \end{aligned}
   
$$

   也是递归函数

6. 函数的最小化操作(minimisation)$\mu$：如果$f$是一个递归函数，那么
   
$$

   \begin{aligned}
   \mu y(f(x,y)=0)=
   \left\{
    \begin{aligned}
    &\text{the least y such that $f(x,z)$ is defined for all $z\leq y$, and f(x,y)=0}\\
    &\text{}\\
    &\text{undefined if otherwise}
    \end{aligned}
    \right.
   \end{aligned}
   
$$

   是一个递归函数。

   即如果存在一个$y_0$使得对任意$z\leq y_0$都有$f(x,z)$有定义，并且$f(x,y_0)=0$，那么$\mu y(f(x,y)=0)=y_0$，否则$\mu y(f(x,y)=0)$发散，即undefined。

   用类似c语言的风格描述，最小化操作相当于一个如下的无界搜索：

   ```code
   int mu(function f, parameter x)
   {
   	int y = 0;
   	while(f(x, y) != 0){
     		y = y + 1;
   	}
   	return y;
   }
   ```

最小化操作十分重要，如果在构造某个递归函数的过程中可以不用最小化操作，我们称所得到的函数为原始递归函数(primitive recursive function)。对一个不是原始递归函数的递归函数，其计算时间复杂度可以非常非常高（甚至比任何指数级复杂度还高），著名的Ackermann函数就是一个不是原始递归的递归函数。

另外，容易发现，如果不引入最小化操作，我们构造出来的递归函数一定是total function，也就是说对应的程序一定会停机。

## 一些常用的递归函数

我们可以通过用以上的归纳定义来构造，证明以下形式的函数都是递归函数。证明并不复杂，碍于篇幅不做介绍，感兴趣的同学可以看参考资料[2]的第二章。这一小节主要作为下一小节的预备知识。

1. 自然数上的加

2. 自然数上变形的减
   
$$

   x\dot{-}y=
   \left\{
   \begin{aligned}
   x-y & \quad x\geq y\\
   0 & \quad \text{otherwise}
   \end{aligned}
   \right.
   
$$


3. 自然数上的乘

4. 阶乘

5. 自然数上的整除 $x/y=\lfloor \frac{x}{y} \rfloor$

6. 整除 
   
$$

   y \vert x=
   \left\{
   \begin{aligned}
   1 & \quad x/y=\frac{x}{y}\\
   0 & \quad \text{otherwise}
   \end{aligned}
   \right.
   
$$

   ​

7. 余数 $x \%y$

8. 大于，小于，等于的特征函数

9. 如果$f_1,f_2,g_1,g_2$都是递归函数，并且对给定的$x$，有$g_1(x),g_2(x)$有且只有一个为0，那么
   
$$

   h(x)=
   \left\{
   \begin{aligned}
   f_1(x) & \quad ,g_1(x)=0\\
   f_2(x) & \quad ,g_2(x)=0
   \end{aligned}
   \right.
   
$$

   是递归函数

10. 如果$f,g$是递归函数那么以下函数是递归函数

$$

   \sum_{z\leq g(k)} f(x)\\
   \prod_{z\leq g(k)} f(x)

$$


11. 如果$f$是递归函数，那么

$$

   \begin{aligned}
   \mu z<y(f(x,z)=0)=
   \left\{
    \begin{aligned}
    &\text{the least z such that $f(x,z')$ is defined for all $z'\leq z< y$, and f(x,z)=0}\\
    &\text{}\\
    &y\quad \text{if otherwise}
    \end{aligned}
    \right.
   \end{aligned}

$$

   注意，这个不同于上文提到最小化操作，这是一个有界搜索，如果在界内没有找到答案，则返回上界。这个操作相当于下列c代码。这个操作不需要最小化操作也能构造，所以如果$f$是原始递归函数，那么对其的有界搜索也是原始递归函数。

```code
   int muy(function f, parameter x, parameter y)
   {
   	int z = 0;
   	while(z < y){
     		if(f(x, z) == 0){
             return z;
     		} else {
             z = z + 1;
     		}
   	}
   	return y;
   }
```

## 构造递归函数的例子*

这一节主要介绍以下递归函数的构造，除了说明它们是递归函数以外，更多的是为了给大家一个感性的认识。大家可以跳过它们的具体构造，但**请记住它们的符号**！特别是$p(x),(x)_y$

- $x​$为素数这一命题的特征函数
- 第$x$个素数$p_x$
- 自然数$x$的质因数分解中，第$y$个素数的指数
- 斐波那契数列

### x为素数的特征函数


$$

prime(x)=
\left\{
 \begin{aligned}
 1&\quad&\sum_{z<x} (z+1)\vert x=2\\
 0&\quad&\text{otherwise}
 \end{aligned}
 \right.

$$


### 第x个素数(从第0个开始数)


$$

\begin{aligned}
p(0) & = 2\\
p(y+1) & = \mu z<(p(y)!+2) (1-prime(z)\cdot(z>p(y)))
\end{aligned}

$$


### x的质因数分解中，第y个素数的指数


$$

(x)_y =\mu z<x(p(y)^{z+1}\vert x)

$$


### 斐波那契数列

先定义

$$

\begin{aligned}
g(0)&=2^03^1\\
g(y+1)&=2^{(g(y))_1}3^{(g(y))_0+(g(y))_1}
\end{aligned}

$$

然后我们有

$$

fib(x)=(g(x))_0

$$



这里用到了将两个自然数编码到一个自然数上的技巧。

# URM程序到自然数的编码（Gödel编码）

这一部分有些繁琐，先说结论，正文看得晕的话不妨跳过。

> 在此，我们完成了对合法URM程序到自然数的一一对应的编码。这就是说，给定一个合法的URM程序，我们可以求出其对应的唯一自然数；给定一个自然数，我们也可以求出其对应的唯一合法URM程序。
>
> 由此，我们可以通过研究自然数的性质来研究URM程序的性质。在下面一节证明URM程序和递归函数的等价性以后，这将意味着我们可以通过研究自然数的性质来研究递归函数的性质。记得我们把递归函数能解决的问题看作“所有我们能解决的数学问题”，这意味着自然数的某些性质就决定了“所有我们能解决的数学问题”！

在前面的介绍中，我们说每一个URM程序都能对应到一个自然数。其实我们能做得更好，每一个合法的URM程序和自然数之间存在着一个一一映射。下面我们会构造出这个映射。**由于我们的目的是给大家直观的感觉，为了阅读的流畅，这一节中会有不严谨和不完整的地方，大家可以在参考资料[2]中找到更为完整的叙述。**

考虑函数

$$

\pi(m,n)=2^m(2n+1)-1

$$

和其反函数

$$

\begin{aligned}
\pi_1(x) &= (x+1)_0\\
\pi_2(x)&=((x+1)/2^{\pi_1(x)}-1)/2
\end{aligned}

$$

不难验证，上述递归函数函数$\pi$完成了两个自然数到一个自然数的一一对应的编码，而$\pi_1,\pi_2$则完成了相应的解码。

更进一步的，我们也可以用类似的方式编码三个自然数，定义

$$

\zeta(m,n,q)=\pi(\pi(m,n),q)

$$

容易看出，我们同样可以写出它的解码函数。

这样，对于每一种URM指令的不同参数，我们都可以得到其对应的自然数。更进一步的，我们通过$mod\ 4$操作来区分四种不同的操作。下面我们直接给出四种操作的自然数编码。记指令p的编码为$e(p)$

1. $e(Z(n))=4n$
2. $e(S(n))=4n+1$
3. $e(T(m,n))=4\pi(m,n)+2$
4. $e(J(i,m,n))=4\zeta(i,m,n)+3$

不难看出，以上给出了每一条URM指令到自然数的一一对应的编码，并且可以有效地被解码。

接下来，我们只要能将任意长度的自然数序列一一对应地编码成一个自然数，就能完成将任意长度的URM程序编码成一个自然数的任务。下面给出这一编码函数$\tau:\cup_k \mathbb{N}^{k}\to \mathbb{N}$。对任意的自然数序列$a_1,...,a_n$

$$

\tau(a_1,...,a_n)=2^{a_1}+2^{a_1+a_2+1}+2^{a_1+a_2+a_3+2}+...+2^{a_1+a_2+...+a_n+n-1}-1

$$

注意，在这里n其实也是$\tau$的参数，我们可以构造出一个递归函数来反求出n，这里不给出具体构造。

在此，我们完成了对合法URM程序到自然数的一一对应的编码。这就是说，给定一个合法的URM程序，我们可以求出其对应的唯一自然数；给定一个自然数，我们也可以求出其对应的唯一合法URM程序。

由此，我们可以通过研究自然数的性质来研究URM程序的性质。在下面一节证明URM程序和递归函数的等价性以后，这将意味着我们可以通过研究自然数的性质来研究递归函数的性质。记得我们把递归函数能解决的问题看作“所有我们能解决的数学问题”，这意味着自然数的某些性质就决定了“所有我们能解决的数学问题”！

这一编码方式称为Gödel编码，哥德尔提出时将其用在谓词逻辑到自然数的编码，细节十分繁琐，但思路很明确，就是归纳定义的字符串到自然数的一一对应。这一编码在哥德尔不完备定理的证明中起到了重要的作用。

# URM程序和递归函数的等价性（Universal程序）

这一部分其实涉及两个方向

1. 给定一个递归函数，我们通过机械的手段总能写出一个计算它的URM程序。
2. 给定一个URM程序，我们总能机械地写出一个递归函数，使得这个URM程序恰好计算了这个递归函数。

但本文将略去第一部分。这首先是因为要明确地写出这一过程很繁琐，其次是因为我相信任何一个接受过基本编程训练的人都能完成这一证明。证明的思路就是，按照递归函数的6个定义，归纳地给出对应的程序构造。

下面我们来介绍，给定一个URM程序，我们如何有效地得到一个计算它的递归函数。这个方法很tricky，我们得到的其实是这样的**一个**递归函数，这个函数的参数有两个，一个是给定URM程序对应的自然数编码，第二个是输入参数的自然数编码，计算的是第一个参数对应的URM程序在输入是第二个参数对应的参数时的输出。

这样，这个递归函数其实可以计算任何一个URM程序在任意输入下的输出。我们称这个“**万能**”的递归函数对应的URM程序为Universal程序，这个程序就像一个虚拟机，它可以模拟一切其他的URM程序，甚至模拟它自身！换句话说，这一个程序就能解决“所有我们能解决的数学问题”。

在我的印象中，Universal程序的存在性是图灵在图灵机上得到的结论，这一结论无疑大大地增加了人们对图灵机的信心。（如果有人考据到这一叙述有误，请联系我改正...）

其实说出了以上思路，相信很多人经过不久的思考都能写出这个“万能”递归函数，下面并没有明确地写出这个函数，而是给出这个Universal程序需要用到的“子函数”和构造过程，这是为了让大家能体会到这个函数确实能写出来。这个函数的具体定义可以在参考资料[2]中找到。

- 定义寄存器的configuration为$c=2^{r_0}3^{r_1}...=\prod_i p_i^{r_i}$，即寄存器的状态编码到的自然数
- $c_n(e,x,t)$：编码为$e$的程序在输入为$x$，运行到第$t$步时，其寄存器的configuration。
- $j_n(e,x,t)$：编码为$e$的程序在输入为$x$，运行到第$t$步时，其运行到的指令的序号。
- $\sigma_n(e,x,t)=\pi(c_n(e,x,t),j_n(e,x,t))$：程序运行到第$t$步时的“状态”。

上面的$n$表示$x$的维度，我们可以考虑$n=1$。这样，我们就有办法完全地描述一个程序的运行状态了，接下来我们要写出在运行一条指令后，程序的状态发生了怎样的变化

下面我们给出更具体的细节

$$

\begin{aligned}
c_n(e,x,0)&=\text{the initial configuration of parameter x}\\
c_n(e,x,t+1) &= 
\left\{
\begin{aligned}
c_n(e,x,t)/p(k)^{(c_n(e,x,t))_k} & \quad & \text{if the $j_n(e,x,t)$th instruction of e is Z(k)}\\
c_n(e,x,t)*p(k) & \quad & \text{if the $j_n(e,x,t)$th instruction of e is S(k)}\\
c_n(e,x,t)/p(j)^{(c_n(e,x,t))_j}*p(i)^{(c_n(e,x,t))_i} & \quad & \text{if the $j_n(e,x,t)$th instruction of e is T(i, j)}\\
c_n(e,x,t) & \quad & \text{if the $j_n(e,x,t)$th instruction of e is J(i, j, k)}
\end{aligned}
\right.\\

j_n(e,x,0)&=0\\
j_n(e,x,t+1) &= 
\left\{
\begin{aligned}
i & \quad & \text{if the $j_n(e,x,t)$th instruction of e is J(i, j, k) and $r_j=r_k$}\\
j_n(e,x,t) & \quad & otherwise
\end{aligned}
\right.

\end{aligned}

$$

实际上，根据递归函数的定义，上面的$c_n,j_n$的定义应该合写成$\sigma_n$的定义，相信大家都能完成这一步。

接下来，再定义几个有用的函数

- $S_n(e,x,y,t)$是以下命题的特征函数：$j_n(e,x,t)=length(e)\text{ and }(c_n(e,x,t))_0=y$即编号e的程序，在输入是x的时候，在t步以内停止，并输出结果y
- $H_n(e,x,t)$是以下命题的特征函数：$ j_n(e,x,t)=length(e)$即编号e的程序，在输入是x的时候，在t步以内停止

注意，根据我们在上一节的描述，编码为e的程序的指令数量是可以由e计算出来的，我们称之为$length(e)$

终于可以写出我们的Universal程序了！

$$

U_n(e,x)=(\mu z(1-S_n(e,x,(z)_0,(z)_1))

$$

这里我们要注意两点。首先，$S_n$的构造中我们并没有用到无界搜索$\mu$，这意味着$S_n$是原始递归的。其次，$U_n$一共就只有一个$\mu$，这意味着任何一个递归函数，都可以仅用一个无界搜索就写出来！

以上，我们证明了递归函数的一个重要性质：存在一个递归函数，它可以计算所有的递归函数。这意味着，我们只需要一套物理设备，给它不同的参数，就能实现所有可能实现的功能！这无疑是现代计算机的坚实基础之一。

在证明了URM模型和递归函数的等价性之后，我们或许可以大胆地称递归函数为**可计算函数**了！

接下来还要介绍几个符号

- $\phi_k(x)$：表示自然数编码为k的递归函数
- $W_k$：表示自然数编码为k的递归函数的定义域
- $E_k$：表示自然数编码为k的递归函数的值域

下文中我们将会常常忽略递归函数的具体形式，只是用它的编码k来代替。需要注意的是，对编码不同的两个URM模型，它们可能计算的是相同的递归函数，即存在$a\neq b,\phi_a=\phi_b$，这是由于我们可以加没用的代码...

我们还要介绍一个很直观定理：s-m-n定理

s如果$f(x,y)$是一个递归函数，那么我们一定可以给出一个total的递归函数$k$，使得$f(x,y)=\phi_{k(x)}(y)$。这里我们不给出证明。

# 不可判定问题（停机问题）

## 一个不可计算的total函数

在URM模型的部分我们提到，存在很多从自然数到自然数的函数，它们是(用递归函数)不可计算的。接下来我们首先尝试找到一个不可计算的函数。考虑以下函数

$$

f(x)=
\left\{
\begin{aligned}
\phi_x(x) + 1&\quad&\text{if $\phi_x(x)$ is defined}\\
0&\quad&\text{if $\phi_x(x)$ is undefined}
\end{aligned}
\right.

$$

我们发现，对任何一个可计算函数$\phi_k$我们都有$f(k)\neq \phi_k(k)$，从而$f\neq \phi_k$。由于$\phi_k$列举出了所有的可计算函数，所以$f$必然不是一个可计算函数！考察$f$的构造过程，实际上我们使用了著名的对角线方法：

|          | 0                    | 1                    | 2                    | 3           | ...  |
| -------- | -------------------- | -------------------- | -------------------- | ----------- | ---- |
| $\phi_0$ | $\phi_0(0)\neq f(0)$ | $\phi_0(1)$          | $\phi_0(2)$          | $\phi_0(3)$ |      |
| $\phi_1$ | $\phi_1(0)$          | $\phi_1(1)\neq f(1)$ | $\phi_1(2)$          | $\phi_1(3)$ |      |
| $\phi_2$ | $\phi_2(0)$          | $\phi_2(1)$          | $\phi_2(2)\neq f(2)$ | $\phi_2(3)$ |      |
| ...      |                      |                      |                      |             |      |

实质上，$f$就是一个与每一个递归函数在对角线上不同的函数，这与证明实数和自然数“不一样多”的方法如出一辙。能使用对角线方法的本质条件是所研究的对象必须是可列的(与自然数集等势)，这使得它在研究对象都是可列的可计算理论中非常的实用。接下来我们可以来讨论停机问题了。

## 停机问题

前面我们对可判定问题的定义如下：如果一个自然数上的谓词$M$的特征函数$c_M$是递归函数，则我们称$M$可判定。那么是否存在一个不可判定的问题呢？

图灵给出了一个这样的问题，即停机问题"$x\in W_y$"。我们只需要考虑一个弱化版的停机问题“$x\in W_x$”，即是否存在这么一个可计算函数，它能判断编码为$x$的递归函数是否在$x$处有定义（或者编码为x的URM模型在输入是x时是否停机）？下面我们用反证法来证明这一问题是不可判定的，从而停机问题"$x\in W_y$"更是不可判定的。

假设$x\in W_x$是可判定的，根据定义，我们知道它的特征函数

$$

c(x)=
\left\{
\begin{aligned}
1&\quad&x\in W_x\\
0&\quad&x\notin  W_x
\end{aligned}
\right.

$$

是可计算的。根据递归函数的定义，我们发现以下函数也是可计算的

$$

g(x)=
\left\{
\begin{aligned}
0&\quad&c(x)=0\\
\text{undefined}&\quad&c(x)=1
\end{aligned}
\right.

$$

换一个写法

$$

g(x)=
\left\{
\begin{aligned}
0&\quad&x\notin W_x\\
\text{undefined}&\quad&x\in W_x
\end{aligned}
\right.

$$

可以得到这样的结论$x\in Dom(g)\iff x\notin W_x$

|          | 0                    | 1                    | 2                    | 3           | ...  |
| -------- | -------------------- | -------------------- | -------------------- | ----------- | ---- |
| $\phi_0$ | $\phi_0(0)\neq g(0)$ | $\phi_0(1)$          | $\phi_0(2)$          | $\phi_0(3)$ |      |
| $\phi_1$ | $\phi_1(0)$          | $\phi_1(1)\neq g(1)$ | $\phi_1(2)$          | $\phi_1(3)$ |      |
| $\phi_2$ | $\phi_2(0)$          | $\phi_2(1)$          | $\phi_2(2)\neq g(2)$ | $\phi_2(3)$ |      |
| ...      |                      |                      |                      |             |      |

考虑这个函数与可计算函数在对角线上的差异，我们可以发现$g$与任意一个可计算函数在对角线上的收敛(或发散)情况都不相同，从而$g$应该是不可计算的。这与$g$是可计算的相矛盾。故我们一开始的假设是错的，$x\in W_x$应该是不可判定的。

**这样来看，停机问题的结论和对角线方法的内涵有着十分重要的联系。**

关于停机问题的一些更有趣的结论，可以参看matrix67的文章《停机问题、Chaitin常数与万能证明方法》：http://www.matrix67.com/blog/archives/901

下面再给出一些不可判定的问题供大家体会

- $\phi_x=\phi_y$
- $\phi_x=0$（常零函数）
- $x\in E_x$

接下来的一节可能会有点莫名其妙，这是因为接下来的很多定义都是为证明哥德尔不完备定理服务的，可能要等到证明时它们的意义才会显现出来。建议大家浏览一下，有一些印象就好了。

# 对自然数子集的分类

## 可判定性，半可判定性

对于任何一个自然数上的谓词$M(x)$，我们总可以把它看成一个自然数的子集$M$，满足这样的性质$x\in M\iff M(x)\text{ is true}$。在上一节我们知道了谓词有可判定与否之分，这样，我们可以用自然数子集对应的谓词的可判定性来给自然数的子集分类。

我们再引入一种可判定性——半可判定。定义如下：对谓词$M(x)$，如果以下函数是可计算的，则我们称$M(x)$是半可判定的：

$$

c_M(x)=
\left\{
\begin{aligned}
1&\quad&M(x)\text{ is true}\\
\text{undefined}&\quad&\text{otherwise}
\end{aligned}
\right.

$$

不难看出，$x\in W_x$是半可判定的，这是因为我们总可以用Universal程序来模拟$\phi_x$，如果其停止了我们就返回1，否则就一直运行下去。

我们有这样的结论

- 如果一个命题是可判定的，则其也是半可判定的。
- 如果一个命题和其否命题都是半可判定的，则该命题是可判定的。直观上，这是因为我们可以轮流执行命题和其否命题的半可判定特征函数，由于一个命题要么是真的要么是假的，两个特征函数总有一个会停止，从而轮流执行的过程也可以停止。至于轮流执行的具体定义，我们可以用$c_n,j_n$来做到，在此不做具体讨论。

那么是否存在不是半可判定的问题呢？"$x\notin W_x$"就是这样的一个问题。由上面的结论，我们知道如果"$x\notin W_x$"是半可判定的，由于"$x\in W_x$"是半可判定的，这将导致"$x\in W_x$"是可判定的，这与我们对停机问题的结论矛盾了。

## 递归集，递归可枚举集

现在我们来对自然数的子集进行分类。对自然数的子集$M$

- 如果$M$对应的谓词$x\in M$是可判定的，则我们称$M$为递归集(recursive set)

  即存在可计算的$c$使得
  
$$

  c(x)=
  \left\{
  \begin{aligned}
  1&\quad&x\in M\\
  0&\quad&x\notin  M
  \end{aligned}
  \right.
  
$$

  ​

- 如果$M$对应的谓词$x\in M$是半可判定的，则我们称$M$为递归可枚举集(recursively enumerable set)，简写为r.e. set

  即存在可计算的$c$使得
  
$$

  c(x)=
  \left\{
  \begin{aligned}
  1&\quad&x\in M\\
  \text{undefined}&\quad&x\notin  M
  \end{aligned}
  \right.
  
$$





我们从直观上看一下这两种集合。对递归集，我们能有效地判断给定自然数x是否在该集合内(用特征函数)，也能从小到大列出其中的元素(按顺序求特征函数在每一个自然数的值)。

对递归可枚举集，我们可以知道一个自然数在里面，但我们无法知道一个自然数不在里面(这一点很奇怪)。另外，我们可以没有遗漏地枚举出其中的元素，虽然这可能需要无限长的时间。我们可以这样做，从$n=0$开始，每次对所有的$k\leq n$，我们都只运行特征函数$c(k)$不多于$n$步，逐渐增加$n$的值，这样，每一个会停机的$c(k)$总会在某一个足够大的$n$上停机，从而对每一个$k\in M$我们总能找到它。这样看，递归可枚举集指的是那些我们能知道哪些自然数在里面，但不能知道哪些自然数不在里面的集合，也就是说我们只能把里面的元素一个一个找出来，没有别的捷径可走。

另外从Universal程序的结论，我们可以知道任何一个可计算函数的定义域$W_x$都是一个递归可枚举集(按枚举递归可枚举集元素的方法得到在定义域中的$x$)，而每一个递归可枚举集也正好是某个可计算函数的定义域(就是它的特征函数)。（其实值域也有相同的结论）

## productive set, creative set

我们称集合$A$是productive的，当且仅当存在total且可计算的函数$g$使得

$$

\forall W_x\subseteq A\Rightarrow g(x)\in A\backslash W_x

$$

$\backslash$表示集合的差。

这一定义有些抽象，我们可以这样直观地看，对一个productive集合$A$，当给定它的任一递归可枚举子集$W_x$时，我们总能在有限的时间内确切地找出一个在$A$中而不在$W_x$中的元素$g(x)$。

我们可以通过取$A=W_x$构造矛盾来证明productive集合一定不是递归可枚举的。

我们称一个递归可枚举集$B$是creative的，当且仅当其关于自然数的补集$\overline{B}$是productive的。

关于creative set和productive set最好的例子就是$K=\{x\vert x\in W_x\}$和$\overline{K}=\{x\vert x\notin W_x\}$.

$\overline{K}$的productive性质可以通过选择$g(x)\equiv x$得到。

## simple set

存在着这样的递归可枚举集，它不是creative的，我们称之为simple set。simple set的构造可见参考资料[2]，这一部分只是为了说明r.e. set并不等于creative set。

# 哥德尔不完备定理

终于到哥德尔不完备定理了！这一部分中，为了思路的清晰，我们会忽略很多的细节，这些细节可以在参考资料中找到。由于这部分和前面不那么连贯，我们先介绍一下这一章的思路。

首先，我们建立陈述(命题)和递归函数的关系。

然后，我们通过这个联系说明所有语义正确的陈述形成的集合是productive的，而所有可以被证明的陈述形成的集合是recursively enumerable的。在前面我们知道productive集合一定不是r.e.的，从而不是所有语义正确的陈述都能被证明。

再然后，我们介绍一个不能被证明的“真”命题。实质上，这个问题就是用自然数和一阶谓词逻辑的语言，构造一个语义是“这个陈述不可证明”的陈述。

接下来，我们尝试抛弃“真”这个直观的概念得到更抽象的结果，说明不可证明陈述的存在只和自然数公理系统的一致性有关，这是哥德尔的结果。

最后，我们介绍Rosser对这一结果的改进。

## 非对即错的陈述(statement)

陈述是谓词逻辑中的公式，命题是陈述的语义（给学过逻辑的同学说一下，看不懂也无所谓）。

我们可以用谓词逻辑对自然数的性质进行一些描述，如“x是素数”，“每个自然数都有唯一的质分解”，这些描述称为陈述（当然，实际上陈述需要用形式化的语言来写，这里用自然语言简化）。当给一个陈述中的所有自由变元都加上$\forall$来约束时，我们会发现，一个陈述要么是真的，要么是假的。这里的“真”和“假”是语义上的，我们可以通过想象把每一个的自然数都带入陈述中被$\forall$约束的变元检验来得到“真”或者“假”的结果。例如$\forall x$"x是素数"，当我们把每个自然数依次带入x时，我们很快就会找到一个不是素数的自然数，从而这一陈述是假的。

当然，由于这样对任何一个包含$\forall$的陈述“真假”的判断都要进行无数次检验，这是不可能办到的，数学家们自然会想，我们能不能给出一组作为公理的陈述，以它们为起点用逻辑规则来生成所有“真”的陈述？

用数学的语言来说，假设

- $S$指的是所有有意义的没有自由变量出现的陈述的集合
- $T$指的是$S$中所有“真”陈述的集合
- $F$指的是$S$中所有“假”陈述的集合

我们有$T\cap F=\emptyset, T\cup F=S$

由于每个陈述都是归纳定义的字符串，我们可以把每个陈述都一一对应地编码到自然数上，从而陈述的集合可以对应到自然数的集合上，我们写成$S=\{\theta_0,\theta_1,\theta_2,...\}$。接下来，我们就可以问这样的两个问题了

1. 集合$T$是否是递归可枚举的？如果答案是肯定的，我们就可以有效地枚举出所有正确的命题了！
2. 是否存在一个$T$的递归的子集，所有$T$中的元素都可以从这个递归子集通过逻辑推出。如果答案是肯定的，我们就可以把这个递归的子集作为公理系统，证明所有"真"的陈述。

然而遗憾的是，哥德尔证明了对于自然数上的陈述，上述两个问题的答案都是否定的。这意味着

1. 我们无法找到(枚举出)所有正确的命题。
2. 我们无法找到一个能证明所有正确命题的公理系统。

注意，我们这里所说的“正确”指的是按照$\forall$的语义，把所有可能取值带入验证得到的正确。而我们通常所说的“正确”还有一个意思，指的是可用公理证明。这两种对“正确”的定义的差异，使我们不得不考虑那些经得起验证，却无法被证明的陈述的存在性。

## 陈述和递归函数的对应

### 哥德尔的引理（在“真”语义下的形式）

对一个特征函数是原始递归的谓词$M(x_1,x_2,...,x_n)$，我们可以通过机械的过程构造一个自然数上含有$n$个变元的陈述$\sigma(x_1,x_2,...,x_n)$，使得对任意$a_1,...,a_n\in \mathbb{N}$，有$M(a_1,...,a_n)$为真当且仅当$\sigma(a_1,...,a_n)\in T$。用我们前面的语言来说，如果$c_M=\phi_n$，那么存在total可计算的$g$，对$\sigma=\theta_{g(n)}$，有$c_M(x_1,...,x_n)=1\iff \sigma(x_1,...,x_n)\in T$

也就是说，原始递归的谓词和自然数上的命题有着对应的关系。这一点可以通过对$c_M$的构造过程进行归纳来证明，具体的证明写起来会很tedious(哥德尔原话)，我们会在最后给出哥德尔版本的引理的证明。

不过我觉得这一点还是很直观的，因为原始递归函数的构造涉及到的操作在自然数上都很直观，写成自然数上的陈述并不会遇到什么障碍。

### Productive的扩展

对自然数上的集合$A,B$，如果A是productive的，且存在total可计算函数$f$使得$x\in A\iff f(x)\in B$，那么$B$也是productive的。这一点的证明可以通过考虑函数$f(g(x))$得到，其中$g(x)$是$A$的productive性质对应的那个函数。

## 不可证明的“真”命题

回忆Universal程序部分的函数

- $H_n(e,x,t)$是以下命题的特征函数：$ j_n(e,x,t)=length(e)$即编号e的程序，在输入是x的时候，在t步以内停止

$H_n$是原始递归的，并且程序$e$会停机当且仅当$\exists z(H_n(e,x,z)=1)$。这样根据哥德尔的引理，我们能构造一个自然数上的陈述$\sigma_R$，使得$H_n(e,x,z)=1\iff \sigma_R(e,x,z)\in T$。那么我们有

$$

e\in K \iff \exists z\ \sigma_R(e,x,z)\in T\\
e\notin K \iff \neg\exists z\ \sigma_R(e,x,z)\in T

$$

假设$H_n(e,x,z)=\phi_k(e,x,z),\sigma_R(e,x,z)=\theta_{f(k)}$，由于我们清楚对$\sigma_R$，应该不难看出我们可以写出可计算的$f'$使得$\neg\exists z\ \sigma_R(e,x,z)=\theta_{f'(f(k))}$，更进一步的，我们可以有效地把$\sigma_R$中的符号$e$和$x$替换成任何自然数$m$，这就是说存在可计算的$f''$使得$\neg\exists z\ \sigma_R(m,m,z)=\theta_{f''(f'(f(k)),m)}$。由于$f'',f',f$都是可计算的，他们的复合也是可计算的，也就是说，我们可以找到可计算函数$g$使得$\neg\exists z\ \sigma_R(m,m,z)=\theta_{g(m)}$。

记

$$

\mathbb{T}=\{x\vert \theta_x\in T\}

$$

那么我们有

$$

\begin{aligned}
n\in\overline{K} & \iff n\notin K\\
& \iff \theta_{g(n)}\in T\\
& \iff g(n)\in\mathbb{T}
\end{aligned}

$$

根据之前提到productive集合的性质，我们有$\mathbb{T}$是productive的，记其productive性质的函数为$g_T$。

由此，我们证明了**所有“真”陈述组成的集合是productive的。**

接下来，我们说明所有可证明的陈述组成的集合是recursively enumerable的。

首先，我们需要这样一个前提：所有公理的下标组成的集合应当是recursive的。这就是说

$$

\mathbb{A}=\{x|\theta_x\text{ is an axiom}\}

$$

是recursive的。这是因为，给定一条陈述，我们必须要有有效的方式判断它是否是公理...

在这样的前提下，我们就可以写出一个万能的证明程序，它可以证明所有能从公理推出来的陈述！

首先我们要说明，判断一个证明是否正确，这一问题是可判定的。

一个合法的证明过程应该是一个有限的陈述序列，满足

1. 其中的每条陈述，要不然是公理，要不然就是用该条陈述前面的陈述通过逻辑公式推出来的。
2. 最后一条陈述是我们要证明的结果。

由于每条陈述都有一个编号，回想哥德尔编码的部分，我们也可以构造陈述的序列与自然数一一对应的编码。

由于公理集合是可判定的，验证证明过程的程序就应当是可计算并且可判定的，这是因为证明合法的每一个条件，都只需要有限的步骤就能做完。

这个程序的具体构造，可以参见哥德尔文章[1]中的$Bw$和$B$函数。

到这里，我们得到了一个可计算函数$LegalProve(x,y)$，其中$x$是一个陈述序列的编码，$y$是一个陈述，这一函数验证$x$是否对应一个合法的$y$的证明。

那么寻找$y$的证明的特征函数，就可以写成$Prove(y)=\mu x(LegalProve(x,y))$，亦即我们搜索所有的陈述序列，直到找到一个陈述序列，其恰好是$y$的证明。

根据我们前面的构造过程，我们知道当$y$是可证明的时候，以上函数$Prove$会停机；当$y$是不可证明的时候，以上函数$Prove$不会停机。

这样，所有可证明的陈述组成的下标集合恰好就是某个递归函数的定义域，根据前面的结论，所有可证明的陈述下标集合是一个recursively enumerable set

$$

Pr=\{x\vert x\text{ is provable}\}=W_{pr}

$$

如果公理系统是"对"的，我们应该有$Pr\subseteq \mathbb{T}$。这也就是说

$$

g_T(pr)\in \mathbb{T}\backslash Pr

$$

即存在一个“真”陈述，其不可证明。

我们可以看到，$g_T(x)$实际上得到的是不满足$W_x$性质的真命题，当我们得以把$W_x$选成"可证明的命题"的时候，实质上我们就完成了"不能被证明的真命题"的构造。

到这里，我们找到一个不可证明的“真”陈述。然而这里还有一点点不严格的地方，那就是我们的证明还依赖着“真”的概念。我们将在下面尝试去掉“真”这一概念。

## 哥德尔的不完备定理（这部分有点问题，维护中！）

先介绍两个概念。

### 一致性(consistent)

我们说一个形式化系统是一致的，当这个形式化系统中不存在一个陈述$\tau$使得以下两个陈述都是可证明的：

$$

\tau,\neg\tau

$$

这就是说，这个系统中不能存在一个陈述，它和它的否定都是可证明的，或者说它即是对的又是错的。这是因为在我们的逻辑中，以下陈述对任意$a,b$都是永真的

$$

(a\land \neg a)\to b

$$

这也就是说，这样一个矛盾的陈述会导致所有的陈述都是对的，从而使得我们的公理系统失去了意义。

### $\omega$-consistent

我们说一个形式化系统是$\omega$-consistent的，当这个形式系统中不存在一个陈述$\tau(y)$使得以下的所有陈述都是可证明的：

$$

\exists y\ \tau(y),\neg\tau(0),\neg\tau(1),\neg\tau(2),...

$$

这就是说，这个系统中不能存在这样一个性质，我们既能证明存在一个自然数满足这个性质，又能证明任意一个自然数都不满足这个性质。一个$\omega$-consistent的系统一定是consistent的，只要取$y$是不存在的自由变量就好了。

### 哥德尔的引理

先介绍哥德尔原文中的一个引理

对一个特征函数是原始递归的谓词$M(x_1,x_2,...,x_n)$，我们可以通过机械的过程构造一个自然数上含有$n$个变元的陈述$\sigma(x_1,x_2,...,x_n)$，使得对任意$a_1,...,a_n\in \mathbb{N}$，有$M(a_1,...,a_n)$为真当且仅当$\sigma(a_1,...,a_n)$是可证明的。用我们前面的语言来说，如果$c_M=\phi_n$，那么存在total可计算的$g$，对$\sigma=\theta_{g(n)}$，有

1. $c_M(x_1,...,x_n)=1\Rightarrow \sigma(x_1,...,x_n)\text{ is provable}$
2. $c_M(x_1,...,x_n)\neq 1\Rightarrow \neg\sigma(x_1,...,x_n)\text{ is provable}$


### 哥德尔的不完备定理

在皮亚诺算数公理系统中，存在一个陈述$\tau$使得

1. 如果皮亚诺算数公理系统是consistent的，那么$\tau$是不可证明的。
2. 如果皮亚诺算数公理系统是$\omega$-consistent的，那么$\neg\tau$是不可证明的。

下面是证明：

根据上一个引理，我们能找出$\sigma$使得

$$

H(e,e,t)=1 \Rightarrow  \sigma(e,t)\text{ is provable}\\
H(e,e,t)\neq 1 \Rightarrow  \neg\sigma(e,t)\text{ is provable}

$$


我们定义两个集合

$$

Pr^*=\{n\vert \exists y\ \sigma(n,y)\text{ is provable}\}\\
Ref^*=\{n\vert \neg\exists y\ \sigma(n,y)\text{ is provable}\}

$$

这两个集合都是r.e.的，这是因为可证明性是半可判定的。

1. 如果$e\in K$，那么存在$m$使得$H(e,e,m)=1$，也就能证明$\exists y\ \sigma(e,y)$。这就是说
   
   $$

   e\in K\Rightarrow \exists y\ \sigma(e,y)\text{ is provable}
   
   $$

   那么我们有$K\subseteq Pr^* $。考虑一致性，我们有$Pr^*\cap Ref^*=\emptyset$，所以$Ref^*\subseteq \overline{K}$。

   记$Ref^*=W_m$，根据$\overline{K}$的productive性质，我们有$m\in \overline{K}\backslash W_m$。

   下面我们考察陈述$\tau\equiv\neg\exists y\ \sigma(m,y)$。如果$\tau$是可证明的，我们就有$m\in W_m$，与前面的结论矛盾了。所以$\tau$是不可证明的。

2. 如果皮亚诺公理系统是$\omega$-consistent的，考虑$\neg\tau=\exists y\ \sigma(m,y)$。

   由于$m\notin K$，我们知道对任意$t$都有$H(m,m,t)\neq 1$，所以$\neg\sigma(m,t)$都是可证明的，这样就有$\neg\tau$是不可证明的。

以上就是哥德尔不完备定理的证明。

## Rosser对哥德尔结果的改进

在这里我只介绍结果，感兴趣的同学可以去查看相关文献。

### Gödel-Rosser Incompleteness Theorem

如果皮亚诺算术公理系统是完备的，那么其中存在一个陈述$\tau$，无论$\tau$还是$\neg\tau$都不是可证明的。

也就是说$\omega$-consistent的条件可以去掉。

# 参考资料

[1] Gödel, Kurt. "Some metamathematical results on completeness and consistency, On formally undecidable propositions of Principia Mathematica and related systems I, and On completeness and consistency." *From Frege to Gödel: A source book in mathematical logic* 1931 (1879).

[2] Cutland, Nigel. *Computability: An introduction to recursive function theory*. Cambridge university press, 1980.

[3] 傅育熙, 上海交通大学, 可计算理论(CS363)教案.
