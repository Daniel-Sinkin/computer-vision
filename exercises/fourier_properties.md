Consider the sine function
$$
s(x) = \sin(2\pi \cdot f \cdot x + \phi_i) = \sin(\omega x + \phi_i)
$$
where
$$
\begin{aligned}
f &= \text{frequency} \\
\omega &= 2\pi \cdot f = \text{angular frequency} \\
\phi_i &= \text{phase} \\
j &= \sqrt{-1} = \text{imaginary number, usually denoted by }i
\end{aligned}
$$
The angular frequency $\omega$ determines how often in one "unit" of time we do a full rotation.

The phase $\phi_i$ determines our starting angle.

Convolving with a linear filter with impulse response $h(x)$ yields
$$
o(x) = h(x) * s(x) = A \sin(\omega x + \phi_o)
$$
where
$$
\begin{aligned}
A &= \text{gain or magnitude} \\
\Delta \phi  := \phi_o - \phi_i &= \text{shift or phase}
\end{aligned}
$$
Using the complex exponential function
$$
\tilde{s}(x) = e^{j\omega x} = \cos(\omega x) + j \sin(\omega x)
$$
yields
$$
\tilde{o}(x) = h(x) * \tilde{s}(x) = A e^{j \omega x + \phi_o}
$$
Qualitatively the Fourier Transforms can be defined as follows:
$$
H(\omega) = \mathcal{F}[h](x) := Ae^{j
 \phi}
$$
i.e. it is the response to a complex exponential of frequency $\omega$ passed through the filter $h(x)$.

This can be expressed as the integral
$$
H(\omega) = \int_{- \infty}^\infty h(x) e^{- j \omega x} dx
$$
or, in the discrete domain, as
$$
H(k) = \frac{1}{N} \sum_{x = 0}^{N - 1} h(x) \exp\left(-j \frac{2\pi k x}{N}\right).
$$
# Properties
See page 135 in [[Computer Vision - Algorithms and Applications - R. Szeliski (2010)]].
## Superposition
$$
\begin{aligned}
\mathcal{F}[f_1 + f_2](\omega) &= \int_{-\infty}^\infty (f_1(x) + f_2(x)) e^{-j \omega x} dx \\
&= \int_{-\infty}^\infty  f_1(x) e^{-j \omega x} +  f_2(x) e^{-j \omega x} dx \\
&= \int_{-\infty}^\infty  f_1(x) e^{-j \omega x} dx +  \int_{-\infty}^\infty f_2(x) e^{-j \omega x} dx \\
&= \mathcal{F}[f_1](\omega) + \mathcal{F}[f_2](\omega).
\end{aligned}
$$
## Shift
Denote by $f_{x_0}$ the function $f$ shifted by $x_0$:
$$
f_{x­_0}(x) := f(x - x_0)
$$
$$
\begin{aligned}
\mathcal{F}[f_{x_0}](\omega) &= \int_{-\infty}^\infty f_{x_0}(x) e^{- j \omega x} dx \\
&= \int_{-\infty}^\infty f(x - x_0) e^{- j \omega x} dx.
\end{aligned}
$$
We can apply the transform
$$
\begin{aligned}
\phi(x) = x + x_0,&&d\phi(x) = 1 dx,\\
\phi^{-1}(-\infty) = -\infty,&& \phi^{-1}(\infty) = \infty,
\end{aligned}
$$
to obtain
$$
\begin{aligned}
\int_{\phi^{-1}(-\infty)}^{\phi^{-1}(\infty)} f(\phi(x) - x_0) e^{-j \omega \phi(x)} d\phi(x) &= \int_{-\infty}^\infty f(x) e^{-j\omega x - j \omega x_0} dx \\
&= e^{-j \omega x_0} \int_{-\infty}^\infty f(x) e^{-j \omega x}dx \\
&= e^{-j\omega x_0} \mathcal{F}[f](\omega).
\end{aligned}
$$
## Reversal
We denote the complex conjugate by:
$$
\overline{a + i b} = a - ib
$$
and note that
$$
\overline{e^{jA}} = \overline{\cos(A) + j \sin(A)} = \cos(A) - j\sin(A) = \cos(-A) + j\sin(-A) = e^{-jA}.
$$
Let $f_-$ be the function $f$ reversed, i.e., defined by:
$$
f_-(x) = f(-x).
$$
$$
\begin{aligned}
\mathcal{F}[f_-](\omega) &= \int_{-\infty}^\infty f_-(x)e^{- j \omega x} dx \\
&= \int_{- \infty}^\infty f(-x) e^{-j\omega x} dx
\end{aligned}
$$
We can apply the transform
$$
\begin{aligned}
\phi(x) = -x,&&d\phi(x) = -dx\\
\phi^{-1}(\infty) = -\infty,&&\phi^{-1}(-\infty) = \infty
\end{aligned}
$$
to obtain
$$
\begin{aligned}
\int_{\phi^{-1}(-\infty)}^{\phi(\infty)} f(-\phi(x))e^{-j\omega\phi(x)}d\phi(x) &= \int_{\infty}^{-\infty} f(x)e^{j\omega x} (-1)dx \\
&= \int_{-\infty}^\infty f(x) e^{j\omega x} dx \\
&= \int_{-\infty}^\infty f(x) \overline{e^{-j\omega x}} dx \\
&= \int_{-\infty}^\infty \overline{f(x) e^{-j \omega x}} dx \\
&= \overline{\int_{-\infty}^\infty f(x) e^{-j\omega x}} \\
&= \overline{\mathcal{F}[f](x)}.
\end{aligned}
$$
As such
$$
\mathcal{F}[f_-] = \overline{\mathcal{F}[f]}.
$$
## Convolution
$$
\begin{aligned}
\mathcal{F}[f * h](\omega) &= \int_{-\infty}^\infty (f*h)(x) e^{-j\omega x} dx \\
&= \int_{-\infty}^\infty \int_{-\infty}^\infty f(t)h(x - t) dt \cdot e^{-j\omega x} dt dx\\
\end{aligned}
$$
Note that
$$
x = t + (x - t)
$$
so we can write
$$
e^{-j\omega x} = e^{-j\omega t} e^{-j\omega (x - t)}
$$
which yields
$$
\begin{aligned}
\int_{-\infty}^\infty \int_{-\infty}^{\infty} f(t)e^{-j\omega t} h(x - t) e^{-j\omega(x - t)} dt dx
\end{aligned}
$$
we can apply Fubini's theorem to this to obtain
$$
\begin{aligned}
\int_{-\infty}^\infty \int_{-\infty}^{\infty} f(t)e^{-j\omega t} h(x - t) e^{-j\omega(x - t)} dx dt
\end{aligned}
$$
which then then be written as
$$
\int_{-\infty}^\infty f(t)e^{-j\omega t} \int_{-\infty}^\infty h(x - t)e^{-j\omega (x - t)} dx dt.\tag{1}
$$
Note that the inner integral can be expressed as
$$
\int_{-\infty}^\infty h(x - t) e^{-j\omega t} e^{j\omega t}dx = e^{j\omega t}\mathcal{F}[h_{-t}](\omega),
$$
applying the shift property yields
$$
e^{j\omega t} e^{-j\omega t}\mathcal{F}[h](\omega) = \mathcal{F}[h](\omega)
$$
Inserting this into $(1)$ gives us
$$
\int_{-\infty}^\infty f(t)e^{-j\omega t}\mathcal{F}[h](\omega)dt = \int_{-\infty}^\infty f(t) e^{-j\omega t}dt \mathcal{F}[h](\omega) = \mathcal{F}[f](\omega) \cdot \mathcal{F}[h](\omega).
$$
## Correlation
## Multiplication
$$
\begin{aligned}
\mathcal{F}[f \cdot g](\omega) &= \int_{-\infty}^\infty f(x) g(x) e^{-j\omega x} dx \\
\mathcal{F}[f] * \mathcal{F}[g] &= \int_{-\infty}^\infty \mathcal{F}[f](t) \mathcal{F}[g](\omega - t) dt
\end{aligned}
$$
## Differentiation
$$
\begin{aligned}
\mathcal{F}[f'](\omega) &= \int_{-\infty}^\infty f'(x) e^{-j\omega x} dx \\
&= [f(x) e^{-j\omega x}]_{-\infty}^{+\infty} - \int_{-\infty}^\infty f(x) \frac{d}{dx} e^{-j\omega x} dx \\
&= - \int_{-\infty}^\infty f(x) (-j\omega)e^{-j\omega x} dx \\
&= j\omega \int_{-\infty}^\infty f(x) e^{-j\omega x} dx = j\omega \mathcal{F}[f](\omega).
\end{aligned}
$$
## Domain Scaling
We only consider the case $a > 0$, for the $a < 0$ case apply the `Reversal` identity.
Denote by $f_a(x) := ax$ the domain scaled version of $f$.
$$
\begin{aligned}
\mathcal{F}[f_a](\omega) &= \int_{-\infty}^\infty f_a(x) e^{-j\omega x} dx \\
&= \int_{-\infty}^\infty f(ax) e^{-j\omega x} dx \\
&= \int_{-\infty}^\infty f(ax) e^{-j\omega ax / a}\frac{1}{a} adx
\end{aligned}
$$
We can apply the transform
$$
\begin{aligned}
\phi(x) = \frac{x}{a},&&d\phi(x) = \frac{1}{a}dx,\\
\phi^{-\infty}(\infty) = \infty,&&\phi^{-\infty}(-\infty) = -\infty.
\end{aligned}
$$
With this we can express the integral as
$$
\int_{\phi^{-1}(-\infty)}^{\phi^{-1}(\infty )} f(\phi(x)) \exp\left(-j \frac{\omega}{a} \phi(x)\right) \frac{1}{a} d\phi(x),
$$
applying the transformation theorem to this yields
$$
\int_{-\infty}^\infty f(x) \exp\left(-j \frac{\omega}{a} x\right) \frac{1}{a} dx = \frac{1}{a} \mathcal{F}[f]\left(\frac{\omega}{a}\right).
$$
## Real Images
## Parseval's Theorem
$$
\sum_{\omega} (\mathcal{F}(f)(\omega))^2 = \sum_{\omega} \left( \sum_{k} f(k) f(\omega - k )\right)^2
$$