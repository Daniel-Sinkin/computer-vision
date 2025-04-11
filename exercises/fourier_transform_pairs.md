# Fourier Transform Pairs
##  Impulse
The following integral is to be understood in a distributional sense:
$$
\mathcal{F}[\delta](\omega) = \int_{-\infty}^\infty \delta(x) e^{-j\omega x} dx = e^{-j\omega \cdot 0} = 1.
$$
## Shifted Impulse
Applying the Known Impulse representation and the `Shift` property yields.
$$
\mathcal{F}[\delta_u](\omega) = e^{-j \omega u}
$$
## Box Filter
Note that the book is slightly wrong on this one, the books formula would work if $\text{box}$ has support $\left[-\frac{1}{2}, \frac{1}{2}\right]$.

$$
\begin{aligned}
\mathcal{F}[\text{box}](x) &= \int_{-\infty}^\infty \text{box}(x) e^{-j\omega x} dx \\
&= \int_{-1}^1 e^{-j\omega x}dx \\
&= \left[\frac{1}{-j\omega} e^{-j\omega x}\right]_{x = -1}^{x = 1} \\
&= -\frac{1}{j\omega} \left(e^{-j\omega} - e^{j\omega}\right) \\
&= -\frac{1}{\omega}  \frac{e^{-j\omega} - e^{j\omega}}{j}.
\end{aligned}
$$
We can write
$$
\begin{aligned}
e^{j\omega} = \cos(\omega) + j\sin(\omega),&&e^{-j\omega} = \cos(\omega) - j\sin(\omega)
\end{aligned}
$$
so that
$$
e^{-j\omega} - e^{j\omega} = \cos(\omega) - j\sin(\omega) - \cos(\omega) - j\sin(\omega) = -2 j \sin(\omega)
$$
With that we get
$$
\mathcal{F}[\text{box}](x) = -\frac{1}{\omega} (-2 \sin(\omega)) = 2 \frac{\sin(\omega)}{\omega} = 2 \operatorname{sinc(\omega)}.
$$
The more general form is obtained by applying the `domain scaling` property.
## Tent
Note that for $|x| \geq 1$ we get $\text{tent}(x) = 0$ for $x \in (0, 1)$ we get
$$
\text{tent}(x) = 1 - x
$$
and for $x \in (-1, 0)$ we get
$$
\text{tent}(x) = 1 + x,
$$
with that we can compute the Fourier transform as follows:
$$
\begin{aligned}
\mathcal{F}[\text{tent}](\omega) &= \int_{-\infty}^\infty \text{tent}(x)e^{-j\omega x} dx \\
&= \int_{0}^1 (1 - x)e^{-j\omega x}dx + \int_{-1}^0 (1 + x)e^{-j\omega x} dx \\
&= \int_{-1}^1 e^{-j\omega x} dx + \left( - \int_0^1 xe^{-j\omega x} dx + \int_{-1}^0 x e^{-j\omega x}\right) \\
&= 2 \text{sinc}(\omega) + \left( - \int_0^1 xe^{-j\omega x} dx + \int_{-1}^0 x e^{-j\omega x}\right) \\
&= ... \\
&= 2 \text{sinc}(\omega) + \left( - 2\text{sinc}(\omega) + 2 \frac{1- \cos(\omega)}{\omega^2}\right) \\
&= 2 \frac{1- \cos(\omega)}{\omega^2} \\
&= ... \\
&= \text{sinc}^2(\omega)..
\end{aligned}
$$
The more general form is obtained by applying the `domain scaling` property.
## Gaussian
$$
\begin{aligned}
G(x; \sigma) &= \frac{1}{\sqrt{2\pi} \sigma} \exp\left(- \frac{x^2}{2\sigma^2}\right) \\
G(\omega; \sigma^{-1}) &= \frac{1}{\sqrt{2\pi} \sigma^{-1}} \exp\left(- \frac{\omega^2}{2(\sigma^{-1})^{2}}\right) = \frac{\sigma}{\sqrt{2\pi}} \exp\left(- \frac{1}{2} \omega^2 \sigma^2\right).
\end{aligned}
$$