# Dynamic VaR Forecasting with GARCH, Machine Learning, and Neural Stochastic Models

## Overview

This project investigates whether increasingly flexible statistical and machine-learning models improve one-day-ahead tail-risk forecasts for an equity portfolio.

I compare classical distributional approaches, GARCH with Student-t innovations, Random Forest volatility forecasting, and neural conditional diffusion models inspired by stochastic differential equations (SDEs).

The models are evaluated out of sample using 99% Value-at-Risk (VaR) forecasts, VaR backtesting, and quantile loss.

A key result is that greater model complexity does not necessarily lead to better tail-risk forecasts. In this experiment, GARCH-t and Random Forest provide the strongest out-of-sample performance, while the neural stochastic models tend to generate overly conservative VaR forecasts.

---

## Research Question

**Can nonlinear machine-learning and neural stochastic models improve one-day-ahead 99% VaR forecasts relative to a classical heavy-tailed GARCH benchmark?**

The project tests the progression

Gaussian distribution  
→ Student-t distribution  
→ GARCH-t  
→ Random Forest  
→ Neural stochastic model

to determine whether modeling heavy tails, time-varying volatility, nonlinear relationships, and learned stochastic dynamics improves portfolio risk forecasting.

---

## Data

The dataset contains daily equity prices from 2016 to 2026 for ten U.S. stocks:

- CAT
- WMT
- CVX
- XOM
- PFE
- JNJ
- BAC
- JPM
- MSFT
- AAPL

Daily log returns are calculated as

$$
r_t = \log\left(\frac{P_t}{P_{t-1}}\right)
$$

and an equally weighted portfolio is constructed:

$$
r_{p,t} = \sum_{i=1}^{N} w_i r_{i,t},
\qquad
w_i = \frac{1}{N}.
$$

The final model comparison uses the 2024–2026 period as the out-of-sample test set.

---

## Methodology

### 1. Historical VaR and Expected Shortfall

Historical 99% VaR is estimated directly from the empirical 1% quantile of portfolio returns.

Expected Shortfall is calculated as the average loss conditional on returns falling below the VaR threshold.

These estimates provide a distribution-free benchmark for observed tail risk.

---

### 2. Gaussian Parametric VaR

Portfolio returns are initially modeled using

$$
r_t \sim N(\mu,\sigma^2).
$$

The Gaussian model substantially underestimates empirical tail risk.

The Normal QQ plot also shows large deviations in the tails, suggesting that the return distribution is not well approximated by a Gaussian distribution.

---

### 3. Student-t Tail Modeling

A Student-t distribution is fitted to portfolio returns to account for heavy tails.

The fitted Student-t distribution provides a substantially better QQ-plot fit and produces VaR and Expected Shortfall estimates closer to the historical estimates.

The analysis therefore suggests that heavy tails are an important feature of the portfolio return distribution.

---

### 4. GARCH(1,1) with Student-t Innovations

Volatility clustering is modeled using

```math
\sigma_t^2
=
\omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2
```

The fitted model produces strong volatility persistence:

$$
\alpha+\beta \approx 0.975.
$$

The autocorrelation of squared standardized residuals is close to zero after fitting the model, suggesting that GARCH captures most of the volatility clustering present in the raw returns.

Student-t innovations are used to account for residual heavy tails.

---

## Machine-Learning Volatility Model

### Random Forest

A Random Forest regression model is used to predict next-day squared returns as a proxy for conditional variance.

Features include:

- current return
- lagged returns
- absolute return
- squared return
- 5-day rolling volatility
- 10-day rolling volatility
- 20-day rolling volatility
- 60-day rolling volatility

The model estimates

```math
\widehat{\sigma}_{t+1}
=
\sqrt{\widehat{r_{t+1}^2}}.
```

An empirical standardized-return quantile estimated from the validation period is then used to construct 99% VaR forecasts.

The Random Forest produces VaR performance very close to GARCH-t in terms of quantile loss.

---

## Neural Stochastic Model

A neural conditional diffusion model is used to learn both conditional drift and conditional volatility:

```math
r_{t+1}
=
\mu_\theta(Z_t)
+
\sigma_\phi(Z_t)\epsilon_{t+1},
```

where \(Z_t\) contains information available at time \(t\).

This specification can be interpreted as a one-step Euler discretization of the stochastic differential equation

```math
dX_t
=
\mu_\theta(Z_t)\,dt
+
\sigma_\phi(Z_t)\,dW_t.
```

Two versions are estimated.

### Gaussian Neural Diffusion

$$
\epsilon_t \sim N(0,1).
$$

The neural network is trained by minimizing Gaussian negative log-likelihood and jointly learns conditional drift and diffusion.

### Student-t Neural Diffusion

The Gaussian innovation is replaced by a standardized Student-t innovation:

$$
\epsilon_t \sim t_\nu.
$$

The degrees-of-freedom parameter is learned jointly with the neural-network parameters.

The estimated value is approximately

$$
\nu \approx 4.0,
$$

indicating substantial heavy-tailed behavior.

However, the Student-t neural model produces excessively conservative VaR estimates and performs worse on quantile loss.

---

## VaR Backtesting

The models are evaluated using several complementary criteria.

### Violation Rate

For a correctly calibrated 99% VaR model,

$$
P(r_t < VaR_t) = 0.01.
$$

Therefore approximately 1% of observations should exceed the predicted loss threshold.

### Kupiec Unconditional Coverage Test

Tests

$$
H_0:
P(r_t < VaR_t)=0.01.
$$

This evaluates whether the overall number of VaR exceedances is correctly calibrated.

### Christoffersen Independence Test

Tests whether VaR violations are independent over time.

A well-specified model should not systematically generate clusters of consecutive violations.

### Conditional Coverage Test

Combines correct violation frequency and violation independence.

### Quantile Loss

Forecast accuracy is also evaluated using the quantile loss function

```math
L_\alpha(r_t,q_t)
=
\left(\alpha - \mathbf{1}_{\{r_t < q_t\}}\right)
(r_t - q_t)
```

with

$$
\alpha=0.01.
$$

Lower quantile loss indicates better tail-quantile forecasts.

---

## Out-of-Sample Results

The final comparison uses 684 observations from the 2024–2026 test period.

| Model | VaR Violations | Violation Rate | Mean Quantile Loss |
|---|---:|---:|---:|
| GARCH-t | 7 / 684 | 1.02% | **0.0002966** |
| Random Forest | 4 / 684 | 0.58% | 0.0002989 |
| Gaussian Neural Diffusion | 2 / 684 | 0.29% | 0.0003198 |
| Student-t Neural Diffusion | 1 / 684 | 0.15% | 0.0004049 |

For the GARCH-t model:

- Kupiec p-value: **0.664**
- Christoffersen independence p-value: **0.076**
- Conditional coverage p-value: **0.189**

All three tests fail to reject the corresponding null hypotheses at the 5% significance level.

The Random Forest achieves a quantile loss very close to GARCH-t, although its violation independence test is more sensitive to a pair of consecutive exceedances.

Both neural models generate substantially fewer violations than the nominal 1% rate, indicating that their VaR forecasts are overly conservative.

---

## Main Findings

### Heavy tails matter

The Gaussian model materially understates observed portfolio tail risk.

Student-t models provide a substantially better representation of extreme returns.

### Volatility is strongly time-varying

Squared returns exhibit substantial serial dependence and volatility clustering.

GARCH(1,1) successfully captures most of this persistence.

### Machine learning is competitive, but not automatically superior

The Random Forest model achieves quantile loss almost identical to GARCH-t despite using a completely different nonlinear modeling framework.

### Greater model complexity does not guarantee better forecasts

The neural stochastic models are more flexible but produce overly conservative VaR estimates.

The Student-t neural diffusion model is particularly conservative despite learning a plausible heavy-tail parameter.

Overall, the results suggest that a relatively parsimonious GARCH-t specification remains difficult to outperform for one-day-ahead portfolio VaR forecasting.

---

## Conclusion

This project demonstrates that improvements in model flexibility do not necessarily translate into improvements in out-of-sample risk forecasting.

Heavy-tailed distributions clearly improve upon Gaussian assumptions, and explicitly modeling volatility dynamics is important for financial returns.

However, increasingly flexible machine-learning and neural stochastic models do not automatically outperform traditional econometric methods.

For this portfolio and test period, GARCH-t provides the strongest overall combination of VaR calibration, violation independence, and quantile-loss performance, while Random Forest produces highly competitive forecasts.

The results highlight the importance of rigorous out-of-sample validation rather than assuming that greater model complexity will necessarily improve financial risk forecasts.

---

## Technologies

- Python
- NumPy
- pandas
- SciPy
- matplotlib
- statsmodels
- `arch`
- scikit-learn
- PyTorch

---

## Repository Structure

```text
dynamic-var-forecasting/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── Combined_dataset.xlsx
│
├── notebooks/
    └── dynamic_var_forecasting.ipynb
