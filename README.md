# Information-Efficient Calibration of the Heston Model

A research project investigating whether Fisher-information-based experimental design can reduce the number of option-market observations required to calibrate a stochastic-volatility model.

The project combines Heston option pricing, parameter calibration, Fisher information, and D-optimal experimental design to investigate how a small, information-efficient subset of the option market can be used for calibration to reduce computational cost.

**Research Question**

Option markets contain many observations across strikes and maturities, but not all observations provide the same amount of information about the underlying model parameters.

This project asks:

Can the Heston model be calibrated accurately using only a small, information-efficient subset of continuously arriving option-market observations?

The investigation has two stages:

- Identify where information is concentrated across the strike–maturity surface using Fisher information.

- Test whether this information can be exploited to reduce the number of observations required for sequential Heston calibration.

## Methodology

### Heston Model

The Heston stochastic-volatility model is used as the underlying pricing and calibration framework. The model parameters are

$$
\theta = (v_0,\kappa,\bar v,\xi,\rho),
$$

where ($v_0$) is the initial variance, ($\kappa$) is the mean-reversion rate, ($\bar v$) is the long-run variance, ($\xi$) is the volatility of variance, and ($\rho$) is the correlation between the asset and variance processes.

Option prices are generated from the model and used as simulated market observations.

### Fisher Information

The Fisher information contributed by each option is calculated from the sensitivity of its price to the model parameters.

This allows the information content of the option market to be mapped across the strike–maturity surface.

The resulting maps show that information is not uniformly distributed: different regions of the option market provide greater information about the Heston parameters than others.

### D-Optimal Selection

The information structure is exploited using a greedy D-optimal subset selection.

For a selected subset (S), the objective is to maximise

$$
\log\det(F_S),
$$

where ($F_S$) is the Fisher information matrix of the selected observations.

This favours subsets that provide strong joint information about the model parameters while avoiding highly correlated or redundant observations.

The resulting subset-size analysis in notebook 04 shows a rapid increase in joint information for the first observations, followed by diminishing marginal information gains. This shows that a relatively small subset can capture a large proportion of the available parameter information.

### Sequential Calibration Experiment

The final experiment in notebook 05 simulates an evolving option market in which the underlying Heston parameters change smoothly over time.

Three observation-selection strategies are compared:

- Fixed Fisher: an information-efficient subset selected once at the beginning.

- Adaptive Fisher: the Fisher-optimal subset is recalculated as the model evolves.

- Random: the same reduced number of observations are selected randomly as a baseline.

The strategies are evaluated using:

- Parameter recovery

The RMSE between the calibrated and true Heston parameters measures how accurately the underlying model is recovered.

- Out-of-sample pricing

The calibrated model is used to price unseen strike–maturity combinations. This tests whether the calibration generalises beyond the observations used for calibration.

- Greeks estimation

Delta, Gamma and Vega are evaluated to investigate whether improvements in parameter estimation and pricing translate into accurate risk sensitivities.

**Key Results**

- Fisher information is spatially structured

The Fisher-information maps show systematic variation across the strike–maturity surface. This indicates that some regions of the option market are more informative about the Heston parameters than others.

- A small subset can retain substantial information

Greedy D-optimal selection rapidly increases the joint Fisher information before reaching diminishing marginal returns. In the simulated market, 20 of 42 available options were sufficient to satisfy the chosen parameter-recovery tolerance of 10%, corresponding to approximately 50% of the full strike-maturity surface of simulated market options.

- Fisher selection outperforms random selection

In the sequential calibration experiment, both Fisher-based strategies achieve lower parameter-estimation and out-of-sample pricing RMSE than random selection using the same number of observations.

For the simulated parameter evolution, adaptive Fisher selection reduced the mean parameter RMSE by approximately 93% and the mean out-of-sample pricing RMSE by approximately 67% relative to random selection.

- The effectiveness of continuous recalibration of the optimal subset is inconclusive

Recalculating the optimal subset as the model evolves requires more computational cost but its improvement in parameter estimation and out-of-sample pricing compared to a fixed initial subset is inconclusive:

approximately ..% improvement in parameter RMSE

approximately -2% improvement in out-of-sample pricing RMSE

The fixed strategy is further chosen as it provides a more computationally efficient approach, requiring only a single subset-selection step.

- Parameter accuracy, pricing accuracy and Greeks are distinct objectives

The experiments also show that accurate parameter recovery does not imply strong performance for every downstream quantity.

The Fisher-selected subset performs within 10% RMSE for parameter estimation compared to calibration using the full options market, while improvements in out-of-sample pricing and Greek estimation are smaller and less consistent, staying within 24% and 17-21% respectively. This highlights an important distinction between parameter identification, pricing accuracy, and risk-measure accuracy.

## Project Structure

derivatives-research/
│
├── notebooks/
│   ├── main/
│   │   ├── 04_fisher_information_selection.ipynb
│   │   └── 05_sequential_calibration.ipynb
│   └── development/
│       ├── 01_black_scholes.ipynb
│       ├── 02_heston_pricing.ipynb
│       └── 03_heston_calibration.ipynb
│
├── src/
│   ├── pricing/
│   └── calibration/
│
├── tests/
│   ├── test_black_scholes.py
│   ├── test_montecarlo.py
│   ├── test_heston.py
│   └── test_heston_cf.py
│
├── pyproject.toml
└── README.md

The notebooks develop the project progressively:

Development and validation: 
Notebook 01: Black–Scholes pricing and implied volatility validation
Notebook 02: Heston pricing and Monte Carlo validation 
Notebook 03: Heston parameter calibration validation

Main experiments:
Notebook 04: Fisher-information mapping and D-optimal option selection
Notebook 05: Sequential calibration and comparison of selection strategies

The reusable pricing, calibration and selection methods are implemented in src/, while the notebooks contain the experiments and analysis. The tests/ folder contains the unit tests for the functions implemented in src/.

## Main Conclusions

The project demonstrates that option-market observations are not equally informative for stochastic-volatility calibration and that this information concentration can be exploited through Fisher-information-based experimental design.

For the simulated market considered here, a carefully selected subset of options can provide better calibration and pricing performance than a random subset of the same size. Furthermore, the initially optimised subset remains effective as the simulated Heston parameters evolve, making a fixed selection computationally cheaper than continuously recomputing the optimal subset.

The framework provides a bridge between statistical experimental design and practical derivatives calibration, while illustrating the distinction in accuracy between Heston parameter identification, pricing accuracy, and risk-measure accuracy.

### Future Work

A natural improvement to the current framework would be to make the selection criterion goal-oriented rather than optimising parameter information alone. For example, the subset could be designed specifically to minimise uncertainty in downstream quantities such as Delta, Gamma or Vega.

Further extensions could investigate:

alternative experimental-design criteria

different market dynamics and parameter trajectories

realistic option surfaces from live data

goal-oriented selection for specific risk measures

## Requirements

The project uses Python and the main scientific dependencies are listed in pyproject.toml.
The notebooks can be run using the project environment after installing the required dependencies.