## Background

### Basic Notations

- $n$th user
- $k$th channel
- $N_k$: total number of users using $k$th channel
- $z^k_n$: additive white Gaussian noise (AWGN)
- $B_{total}$: total bandwidth
- $K$: total number of channels
- $N$: total number of users
- $B_c=B_{total}/K$: divided bandwidth by channels

#### Channel to Noise Ratio (CNR)

CNR is a ratio between the power of the carrier signal (channel) and the power of the noise.

$$
\begin{align}
\Gamma_n^k &= g^o_{n, k} \\
&= \frac{|h^k_n|^2}{\sigma^2_{z_k}} \\
\end{align}
$$

- $h^k_n$: channel response between BS and $n$th user which considers both the path loss $\mathcal{P}_L$ and shadowing effect $h'_{n, k}$ (= Rayleigh fading?)
- $\sigma^2_{z_k}$: variance of AWGN

#### Signal to Noise plus Interference Ratio (SINR)

> NOTE: An assumption behind the following equations is;
>
> $$
> \Gamma^k_1 > ... > \Gamma^k_n > ... > \Gamma^k_{N_k}.
> $$
>
> This means that the 1st user and the last user has the strongest and lowest signal power respectively. According to NOMA protocol, users with lower CNR will be assigned with more power.
>
> $$
> p^k_1 < p^k_2 < ... < p^k_n < ... < p^k_{N_k},
> $$
>
> where $p^k_n$ denotes transmit power of user $n$ using channel $k$.
>
> Due to the characteristic of successive interference cancellation (SIC), an user will treat less power as an interference and decode signals of more power.

$$
\begin{align}
\gamma_{n}^{k} &= \frac{p_n^k \Gamma_n^k}{1 + \sum_i^{n-1}{p_i^k \Gamma_n^k}} \\
&= \frac{P_{n, k}(t) \mathcal{P}_L(d)|h'_{n, k(t)}|^2}{n^2_0 + \sum_{i=1}^{n-1}{P_{n, k}(t) \mathcal{P}_L(d)|h'_{n, k(t)}|^2}}
\end{align}
$$

- $d$: distance between BS and user $n$ which is using channel $k$
- Typically, $n^2_0$ and $1$ represents a constant noise term.
- Equation (2) is from `TPPD` paper.

Overall, SINR is to calculate the ratio between received power (numerator) and other noises and interferences (denominator) which comprises noises ($1$ or $n^2_0$) and interferences (sum of other users’ received power).

Take the numerator, by multiplying the allocated power $p^k_n$$p^k_n \Gamma^k_n$), this results a signal power considering the noise.

And now let’s take a look at the denominator. The 1 denotes the noise and $\sum_i^{n-1}p^k_i\Gamma^k_n$ is to sum all the signal powers which are less than $p^k_n$ (take a look at the NOTE assumption).

By dividing the above two, ratio of signal power versus noises can be calculated.

#### Data Rate

$$
R^k_n(\Gamma^k_n, p^k_1, ..., p^k_n) = B_c\log_2{(1+\frac{p^k_n \Gamma^k_n}{1+\sum_{i=1}^{n-1}p^k_i\Gamma^k_n})}.
$$

- $\frac{p^k_n \Gamma^k_n}{1+\sum_{i=1}^{n-1}p^k_i\Gamma^k_n}$: As described above, this term represents SINR.
- $1 + \frac{p^k_n \Gamma^k_n}{1+\sum_{i=1}^{n-1}p^k_i\Gamma^k_n}$: By adding 1 to SINR, it can prevent logarithm taking zero.
- $\log_2{(1+\frac{p^k_n \Gamma^k_n}{1+\sum_{i=1}^{n-1}p^k_i\Gamma^k_n})}$: The term calculates the achievable data rate for a noisy channel and it’s derived from Shannon capacity formula.
- $B_c\log_2{(1+\frac{p^k_n \Gamma^k_n}{1+\sum_{i=1}^{n-1}p^k_i\Gamma^k_n})}$: Shannon capacity formula, providing an upper bound on the achievable data rate for a given channel. By multiplying the bandwidth (Hz) and achievable data rate, data rate for the given bandwidth is calculated.

According to the above, two users’ data rate can be derived as such:

$$
\begin{align}
R^k_1(\Gamma^k_1, p^k_1, p^k_2) &= B_c\log_2{(1+p^k_1\Gamma^k_1)} \\
R^k_2(\Gamma^k_2, p^k_1, p^k_2) &= B_c\log_2{(1+ \frac{p^k_2\Gamma^k_2}{1+p^k_1\Gamma^k_2})}.
\end{align}
$$
