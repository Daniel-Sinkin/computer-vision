Recall that we if have a vector
$$
\begin{pmatrix}a\\b\\c\end{pmatrix} \in \mathbb{R}^3
$$
then we can extract the coefficients of this vector using the dot product with the standard basis
$$
\left\{\begin{pmatrix}1\\0\\0\end{pmatrix}, \begin{pmatrix}0\\1\\0\end{pmatrix}, \begin{pmatrix}0\\0\\1\end{pmatrix}\right\}
$$
to get:
$$
\begin{aligned}
\begin{pmatrix} a \\ b \\ c \end{pmatrix} \bullet \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix} = a, \\
\begin{pmatrix} a \\ b \\ c \end{pmatrix} \bullet \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix} = b, \\
\begin{pmatrix} a \\ b \\ c \end{pmatrix} \bullet \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix} = c. \\
\end{aligned}
$$

Let
$$
f(x) = A \sin(x) + B\sin(3x) + C \sin(5x).
$$
Recall this means that we have added 3 sine waves with:
* Amplitudes equal to 1, 0.5, 0.2
* Frequencies equal to $1, 3, 5$.
We can interpret $\{\sin(x), \sin(3x), \sin(5x)\}$ as a basis and with respect to that basis we can write the functions as follows:
$$
f = \begin{pmatrix} A \\ B \\ C \end{pmatrix}
$$
In that basis we also get
$$
\begin{aligned}
\sin(x) = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix},&&
\sin(3x) = \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix},&& 
\sin(5x) = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}, 
\end{aligned}
$$
We can interpret the following expression as a type of dot product
$$
f \bullet g := \frac{1}{\pi} \int_{-\pi}^\pi f(x) g(x)dx
$$
and then we can see that
$$
\begin{aligned}
f \bullet (\sin(3x)) &= (A \sin(x) + B \sin(3x) + C \sin(5x)) \bullet \sin(3x) \\
&= A \underbrace{(\sin(x) \bullet \sin(3x))}_{= 0} + B \underbrace{(\sin(3x) \bullet \sin(3x))}_{= 1} + C\underbrace{(\sin(5x) \bullet \sin(3x))}_{=0} \\
&= B.
\end{aligned}
$$
where we have used the fact that
$$
\frac{1}{\pi}\int_{-\pi}^\pi \sin(ax) \cdot \sin(bx) = \begin{cases}1&&a = b\\0&&a \neq b\end{cases}
$$
to obtain
$$
\begin{aligned}
\frac{1}{\pi}\int_{-\pi}^\pi f(x) \sin(x)dx &= A,\\
\frac{1}{\pi}\int_{-\pi}^\pi f(x) \sin(3 x)dx &= B,\\
\frac{1}{\pi}\int_{-\pi}^\pi f(x) \sin(5 x)dx &= C
\end{aligned}
$$
## Going to the complex case
Recall that we can write
$$
e^{i\omega x} = \cos(\omega x) + i \sin(\omega x).
$$
For this we have a similar identity to before:
$$
\frac{1}{2\pi}\int_{-\pi}^\pi e^{i\omega x} e^{-i\eta x}dx = \begin{cases} 1,&&\omega = \eta\\0,&&\text{else}\end{cases}
$$
So if we write
$$
f(x) = A e^{ix} + Be^{3ix} + C e^{5ix} + D
$$
then we can extract the amplitudes $A, B, C$ as before:
$$
\begin{aligned}
A &= \frac{1}{2\pi} \int_{-\pi}^\pi f(x) e^{-ix}dx,\\
B &= \frac{1}{2\pi} \int_{-\pi}^\pi f(x) e^{-i 3x}dx,\\
C &= \frac{1}{2\pi} \int_{-\pi}^\pi f(x) e^{-i 5x}dx,\\
D &= \frac{1}{2\pi} \int_{-\pi}^\pi f(x) dx.
\end{aligned}
$$
Recall the definition of the Fourier transform:
$$
\mathcal{F}[f](\omega) = \frac{1}{2\pi}\int_{-\pi}^\pi f(x)e^{-i\omega x}dx
$$
and note that then
$$
\begin{aligned}
\mathcal{F}[f][1] = A, \\
\mathcal{F}[f][2] = B, \\
\mathcal{F}[f][3] = C, \\
\mathcal{F}[f][0] = D.
\end{aligned}
$$