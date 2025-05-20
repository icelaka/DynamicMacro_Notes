# Decentralized Problem

## Baseline model

The household's problem is:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}\underset{\lbrace%20c_t%20\rbrace_{t=0}^{\infty}%20,\lbrace%20a_{t+1}%20\rbrace_{t=0}^{\infty}}{\max\;}\mathbb{E}_0\sum_{t=0}^{\infty}\beta^t\frac{c_t^{1-\sigma}}{1-\sigma}\;\textrm{,}" />
</p>

subject to

$$
c_t + a_{t+1} = (1 + r)a_t + w l_t e^{z_t}
$$

$$
z_{t+1} = \mu + \rho z_t + \epsilon_{t+1}
$$

$$
\begin{array}{l} 
c_t > 0~, \\
a_t \ge \phi ~, 
\end{array}
$$

$$
a_t ~\textrm{is}\;\textrm{given}~,
$$

where $w$ is the wage rate that households take as given, $|\rho| < 1$, $\epsilon_{t+1} \sim \mathcal{N}(0, \sigma_{\epsilon}^2)$, and the constraints hold for $t=1,2,\ldots$. The firm's problem is similar:

$$
\underset{k_t, l_t}{\max} \; k_t^{\alpha} l_t^{1-\alpha} - w l_t - r_t k_t
$$

$$
\textrm{s.t.}~k_{t+1} = (1-\delta)k_t + i_t
$$

## Baseline model + Financial constraint

By adding in a financial constraint, the household's budget constraint now also includes dividends:

$$
c_t + a_{t+1} = (1 + r)a_t + w l_t e^{z_t} + d_t
$$

where $d_t$ is the dividend. The firm's problem is now:

$$
\max_{k_t, l_t, i_t} k_t^{\alpha} l_t^{1-\alpha} - w k_t = r k_t
$$

$$
\textrm{s.t.}~k_{t+1} = (1 - \delta)k_t + i_t
$$

$$
i_t \leq \phi \pi_t
$$

where $\pi_t = k_t^{\alpha} l_t^{1-\alpha} - w l_t - r k_t$ is the profit of the firm.

Dividends paid to households are equal to firm's profit:

$$
d_t = \pi_t
$$

## Baseline model + Financial constraint + Productive idea

Then, by introducing productive idea, households now have an additional state variable: idea quality ($q$).

Households can invest in idea $i_t$ at a cost, which includes:

- Linear investment cost ($\kappa_0 i_t$)
- Quadratic investment cost ($\kappa_1 i_t^2$)
- Maintenance cost (proportional to the idea quality $\kappa_2 q_t$)

Ideas evolve according to the following process:

$$
q_{t+1} = (1 - \delta_q) q_t + \theta i_t^{\eta} + \epsilon_t
$$

where $\epsilon_t$ is a random shock to the idea quality.

Households have the choice to abandon ideas at any time. There are 2 possible scenarios:

1. If ideas are maintained, income includes stochastic firm profit enhanced by idea quality.
2. If ideas are abandoned, income reverts to the baseline model.
