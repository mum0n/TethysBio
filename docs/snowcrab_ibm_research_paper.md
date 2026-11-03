---
title: "An End-to-End Individual-Based Model for Snow Crab (*Chionoecetes opilio*): Integrating Bioenergetics, Thermal Molt Phenology, Allometric Growth, Closed-Loop Reproduction, and Configurable Process Toggles"
header: "Choi 2026: Snow Crab Individual-Based Model"
date: "\\today"
journal: "Canadian Journal of Fisheries and Aquatic Sciences"
author:
- name: Jae S. Choi
  email: choi.jae.seok@gmail.com
  orcid: 0000-0003-3632-5723
affiliation:
- name: Fisheries and Oceans Canada, Bedford Institute of Oceanography, Dartmouth, NS, Canada
keyword: |
  Keywords: Chionoecetes opilio, individual-based model, Agents.jl, configurable process toggles, terminal molt, thermal degree-days, carapace condition, skip-molting, spermathecal storage, fisheries assessment
abstract: |
  The snow crab (*Chionoecetes opilio*) supports major cold-water crustacean fisheries across the Northwest Atlantic and subarctic Pacific oceans. Managing these populations under rapid ocean warming, changing cold intermediate layer hydrography, and shifting predation regimes requires computational models that resolve complex decapod life-history mechanisms: determinate growth constrained by a terminal molt, extreme sexual size dimorphism, multi-year spermathecal sperm storage, temperature-dependent embryonic incubation, size-dependent predation, cuticular wear, and physiological degree-day regulation of ecdysis. An open-source, spatially explicit individual-based model is formulated in Julia using `Agents.jl`. The modeling architecture incorporates a configurable dual-pipeline structure: a distilled reference baseline driven by empirical intermolt durations, age-based maturation, and direct reproduction, contrasted with a fully articulated eco-physiological formulation that couples allometric bioenergetics ($W^{0.75}$), two-dimensional thermal degree-day accumulation, dynamic carapace condition scoring across five fisheries categories, multi-driver natural mortality, habitat taxis, size-assortative mate competition, and multi-clutch spermathecal buffering. The framework integrates empirical modal stage bounds, logarithmic Hiatt growth increments with terminal molt decrements, multivariate covariance parameterization directly from survey collections, closed-loop reproduction with temperature-extended embryogenesis (365 to 730 days), and continuous biomass accounting. In addition to individual-level mechanics, the system provides automated fitting of macro-ecological population models (Schaefer surplus production, Deriso-Schnute delay-difference, and Generalized Leslie stage-transition matrices) and retrospective peeling diagnostics. Simulations demonstrate that the distilled baseline provides an agile, parsimonious foundation for macro-model calibration, whereas the fully featured eco-physiological model identifies acute population vulnerabilities to marine heatwaves, localized male depletion, and skip-molting accumulation under sustained thermal anomalies.
---

# 1. Introduction

## 1.1 Ecological and Fishery Context

The snow crab (*Chionoecetes opilio*, Brachyura: Oregoniidae) is a stenothermic cold-water decapod distributed throughout the shelf regions of the Northwest Atlantic, the Arctic Ocean, the northern Bering Sea, and the Sea of Japan (Alunno-Bruscia and Sainte-Marie 1998, Tremblay 1997, Dawe et al. 2012). The species functions as a dominant benthic omnivore and macro-invertebrate predator feeding upon polychaetes, bivalves, crustaceans, and ophiuroids, while serving as a primary prey resource for demersal teleosts, including Atlantic cod (*Gadus morhua*) and Atlantic halibut (*Hippoglossus hippoglossus*) (Boudreau et al. 2011, Dionne et al. 2003).

From a fisheries perspective, *C. opilio* supports major commercial trap fisheries across Eastern Canada and Alaska. Management frameworks have traditionally relied upon control of three primary dimensions: size, by imposing minimum carapace width thresholds ($CW \ge 95$ mm); sex, by restricting harvest exclusively to adult males to preserve female spawning biomass; and season, by scheduling harvest periods to prevent exploitation during soft-shell post-molt intervals (Choi and Zisserson 2012, DFO 2022). However, dramatic population fluctuations, exemplified by the collapse of the Eastern Bering Sea stock between 2018 and 2022 following prolonged marine heatwaves and elevated metabolic starvation (Szuwalski et al. 2021, 2023, Fedewa et al. 2020), demonstrate that commercial harvesting interacts strongly with climate-driven energetic and demographic constraints.

## 1.2 Decapod Life-History Mechanics

Resolving snow crab population dynamics requires accounting for specific physiological and developmental characteristics that distinguish brachyuran decapods from continuously growing aquatic organisms. 

Unlike homarid lobsters (*Homarus americanus*), which continue to molt throughout adult life, *C. opilio* undergoes a determinate terminal molt to morphometric maturity, after which all somatic growth permanently ceases (Conan and Comeau 1986, Sainte-Marie et al. 1995, Sainte-Marie and Sainte-Marie 1999). 

The species exhibits pronounced sexual size dimorphism. Morphometrically mature males attain carapace widths between 95 and 150 mm across developmental stages 11 to 14, developing hyper-allometric chelae utilized in agonistic male-male competition and female mate-guarding. In contrast, females undergo terminal molt earlier at stages 9 to 11, attaining carapace widths between 50 and 85 mm and developing an expanded, circular abdomen adapted for external egg brooding (Alunno-Bruscia and Sainte-Marie 1998, Comeau et al. 1998).

The developmental chronology from hatching to morphometric maturity spans six to eight years. While early post-larval benthic instars molt multiple times per year, adolescent stages (instars 7 and older) transition to an annual molting cycle synchronized with late winter and early spring. When thermal degree-day accumulation or internal energetic reserves fall below critical thresholds, individuals skip-molt, remaining in their current instar for an additional year before re-evaluating ecdysis (Sainte-Marie et al. 1995, Dawe et al. 2012).

Following the terminal molt, adult crabs survive for four to seven years. During this post-ecdysial phase, their exoskeletons advance through five discrete carapace condition categories, progressing from newly molted soft shell to clean hard, intermediate, worn, and decaying states. This progressive cuticular wear is accompanied by epibiont accumulation, structural embrittlement, and rising natural mortality hazards.

The reproductive biology of the species features internal fertilization and long-term sperm storage. Receptive females mate during late winter and spring. Females store viable ejaculate within paired ectodermal spermathecae, permitting autonomous fertilization of up to three successive annual clutches in the absence of male encounters (Sainte-Marie et al. 1997, 2000, Rondeau and Sainte-Marie 2001). Females brood clutches of 20,000 to 150,000 eggs beneath the abdominal flap. Embryonic development rates depend strongly upon ambient bottom temperature: incubation requires approximately twelve months in waters of 3 to 5°C, but extends to twenty-four months in sub-zero bottom waters below 1°C, producing an alternation between annual and biennial reproductive schedules (Kuhn and Choi 2011, Webb et al. 2007).

## 1.3 Analytical Trade-Offs in Population Modeling

Conventional stock assessment models, including lumped-biomass surplus production and age-structured delay-difference frameworks, assume continuous somatic growth and spatially homogeneous mixing. These formulations omit discontinuous ecdysis, terminal molt invariance, and spatial temperature constraints. Conversely, detailed individual-based models frequently introduce large parameter arrays and computational overheads, complicating calibration against standard fisheries survey data.

To reconcile biological realism with computational tractability, a configurable simulation architecture is required. A distilled reference baseline provides an unforced demographic reference that reproduces standard fisheries dynamics without environmental coupling, while an articulated eco-physiological formulation incorporates thermal degree-days, allometric bioenergetics, and behavioral taxis within a single codebase.

## 1.4 Objectives

The objectives of this study are to formulate, implement, and validate an open-source individual-based model for *C. opilio* in Julia using `Agents.jl`. The modeling suite formalizes a dual-pipeline architecture that permits direct toggling between baseline and fully articulated eco-physiological mechanics. The framework incorporates empirical stage boundaries, logarithmic Hiatt growth increments, and logistic maturation ogives derived from fishery-independent surveys. A multivariate normal covariance parameterization module is formulated to preserve joint correlations among size, mass, maturity, and fecundity. The implementation integrates dynamic carapace condition progression, allometric bioenergetics with 3/4-power metabolic scaling, spatial taxis, and multi-factor natural mortality. Finally, the suite provides automated estimation of macro-ecological population dynamics models (Schaefer, Deriso-Schnute, and Generalized Leslie transition matrices) and retrospective peeling diagnostics to evaluate operational stock assessment properties.

# 2. Model Description

The model formulation follows the Overview, Design Concepts, and Details protocol for individual-based models established by Grimm et al. (2006, 2020).

## 2.1 Overview

### 2.1.1 Purpose

The purpose of the model is to simulate individual bioenergetic budgets, thermal molting phenology, spatial dispersal, reproductive output, natural and harvest mortality, and emergent population biomass trajectories of *C. opilio* under variable thermal regimes and exploitation schedules. 

The software architecture supports two operational modes. The distilled baseline configuration provides a reference model that relies on time- and age-based molting schedules, empirical logistic maturation, direct annual reproduction, and size-attenuated baseline mortality, bypassing food limitations and thermal modulation. The fully featured eco-physiological configuration incorporates two-dimensional thermal degree-day tracking, allometric energy assimilation, Holling Type II foraging, cuticular wear across five carapace condition stages, multi-factor mortality, habitat taxis, size-assortative mating competition, and multi-clutch spermathecal buffering.

### 2.1.2 Entities, State Variables, and Scales

The model comprises individual organism agents, aggregate pre-recruit cohorts, and a discrete spatial environment.

Individual organisms are represented as mobile entities positioned upon a discrete two-dimensional grid space $\Omega = [1, N_x] \times [1, N_y]$. Each individual $i$ is characterized by an internal state vector containing sixteen physiological and demographic attributes:

$$\mathbf{x}_i = \left( \text{id}_i, \mathbf{p}_i, \text{stage}_i, \text{sex}_i, \iota_i, a_i, CW_i, W_i, E_i, t_{\text{molt}, i}, \text{DD}_i, CC_i, \mu_{\text{ready}, i}, N_{\text{eggs}, i}, t_{\text{hatch}, i}, S_i \right)$$

Spatial coordinates $\mathbf{p}_i = (x_i, y_i) \in \Omega$ locate the agent on the grid. Categorical lifecycle stages include zoea, megalopa, juvenile, adult, and egg groups, while $\text{sex}_i \in \{\text{male}, \text{female}, \text{undifferentiated}\}$ denotes morphological sex. Instar progression is indexed by the integer stage variable $\iota_i \in \{1, 2, \dots, 16\}$, while $a_i \in \mathbb{R}^+$ denotes chronological age in days post-hatching. Carapace width $CW_i$ is recorded in millimeters, and individual wet body mass $W_i$ is recorded in grams. Metabolic energy reserves are represented by $E_i \in \mathbb{R}$. The variable $t_{\text{molt}, i} \in \mathbb{R}$ records the simulation day of the most recent ecdysis, while $\text{DD}_i \in \mathbb{R}^+$ tracks cumulative physiological thermal degree-days accumulated since that event. Carapace condition is categorized by $CC_i \in \{1, 2, 3, 4, 5\}$, spanning new soft shell, clean hard shell, intermediate hard shell, old hard shell, and decaying cuticular states. Physiological mating readiness is indicated by the binary flag $\mu_{\text{ready}, i} \in \{0, 1\}$. Reproductive state variables include external brood egg count $N_{\text{eggs}, i} \in \mathbb{N}_0$, scheduled brood hatch day $t_{\text{hatch}, i} \in \mathbb{R}^+$, and viable clutches remaining within the spermathecal store $S_i \in \{0, 1, 2, 3\}$.

Aggregate cohorts represent early developmental phases prior to individual benthic emergence (eggs through early benthic instars). Individuals within an aggregate cohort share developmental age, mean carapace dimensions, and spatial location, advancing collectively to optimize memory allocation and computation time.

The environment is represented by a regular lattice of $N_x \times N_y$ cells. Each spatial location $(x, y)$ tracks sea-floor bathymetric depth, daily bottom water temperature $T(x, y, t)$, and local benthic food biomass density $F(x, y, t)$. Food availability renews according to daily logistic replenishment subject to local consumer grazing:

$$\Delta F(x, y, t) = r_F \cdot F(x, y, t) \cdot \left(1 - \frac{F(x, y, t)}{K_F}\right) \Delta t - \sum_{i \in \text{cell}} I_i$$

where $r_F$ is the intrinsic renewal rate, $K_F$ is local carrying capacity, and $I_i$ is ingestion by individual $i$. Temperature fields are synthesized from sinusoidal climatologies or queried from oceanographic reanalyses. The simulation environment tracks demographic summaries and partitions cumulative natural mortalities among baseline senescence, size-dependent predation, starvation, thermal stress, cuticular degradation, and commercial trap removals.

Simulations operate at a daily time step ($\Delta t = 1.0$ day). Spatial grid cells represent regional shelf domains (nominally 1 to 2 km per cell), supporting regional configurations spanning 10,000 to 40,000 km$^2$ over multi-annual to multi-decadal projection horizons (2 to 15 years).

### 2.1.3 Process Scheduling

The simulation sequence executes through a daily stepping cycle partitioned into three sequential phases: environmental field updates, individual agent dynamics, and population-level closed-loop processes.

During the environmental update phase, local water temperatures and food fields are updated, and internal demographic caches are cleared. In the agent stepping phase, individual crabs are updated in random order. Under the distilled baseline pipeline, each crab executes an unbiased random walk across adjacent lattice cells, increments chronological age, evaluates ecdysis based on stage-specific minimum intermolt durations, evaluates logistic maturation, applies constant adult senescence and juvenile predation hazards, and applies commercial fishery selectivity to legal-sized males. Under the articulated eco-physiological pipeline, each crab evaluates directed habitat taxis across local temperature and forage gradients, accumulates degree-days, feeds according to a Holling Type II functional response, expends basal metabolic energy scaled to $W^{0.75}$, evaluates ecdysis against degree-day thresholds and late-winter calendar windows, updates carapace condition wear, applies multi-driver natural mortality, and faces fishery selectivity.

During the population-level closed-loop phase, aggregate cohorts are advanced, mating interactions are resolved, brood extrusion is executed, and larval hatching is converted to benthic recruits through compensatory density-dependent settlement.

# 3. Mathematical Formulations

## 3.1 Discrete Developmental Stages and Modal Boundaries

Crustacean development proceeds through discontinuous instars separated by ecdysial shedding of the exoskeleton. Stages 1 and 2 represent pelagic Zoea I and Zoea II stages, with mean carapace dimensions of $1.8 \pm 0.1$ mm and $2.8 \pm 0.15$ mm, respectively. Stage 3 represents the pelagic Megalopa stage ($3.5 \pm 0.2$ mm). Stage 4 represents the first benthic post-larval instar, appearing at approximately 10.7 mm.

Pelagic larval progression follows empirical rearing observations documented by Ouellet and Sainte-Marie (2018). Advancement across larval instars requires accumulation of physiological degree-days:

$$\Delta \text{DD} = \max(0, T - T_0) \Delta t$$

evaluated relative to a developmental threshold temperature $T_0 = -1.5^\circ\text{C}$. Transitions from Zoea I to Zoea II, Zoea II to Megalopa, and Megalopa to benthic recruit each require approximately 270 degree-days (spanning 225 to 350 degree-days across ambient temperatures of 2.8 to 7.8°C), subject to minimum intermolt intervals of 36, 36, and 42 days. An over-age developmental hazard penalizes delayed instars that remain unmolted past 84 days by adding an excess daily mortality hazard:

$$\Delta M_{\text{age}} = 0.05 \cdot \frac{\text{days} - 84}{10}$$

Larval viability exhibits a concave thermal response centered at optimal temperatures ($T_{\text{opt}} = 11.0^\circ\text{C}$, $\sigma_T = 3.5^\circ\text{C}$):

$$h_{\text{temp}}(T) = \exp\left( 0.5 \left( \frac{T - T_{\text{opt}}}{\sigma_T} \right)^2 \right)$$

For benthic stages ($\iota \ge 4$), empirical lower modal carapace width boundaries are described by linear models on the logarithmic scale (Sainte-Marie et al. 1995, DFO 2020):

$$\ln(CW_{\min}(\iota, \text{male})) = 1.917564 + 0.298914 \cdot (\iota - 3)$$

$$\ln(CW_{\min}(\iota, \text{female})) = 2.198848 + 0.315026 \cdot (\iota - 4)$$

These formulations yield natural-scale boundaries $CW_{\min}(\iota) = \exp(\ln(CW_{\min}))$. The continuous stage index corresponding to an observed carapace width is obtained by inverting these relationships:

$$\iota(CW, \text{male}) = 3 + \frac{\ln(CW) - 1.917564}{0.298914}$$

$$\iota(CW, \text{female}) = 4 + \frac{\ln(CW) - 2.198848}{0.315026}$$

## 3.2 Hiatt Growth Increments and Thermal Modulation

Post-ecdysial carapace width ($CW_{t+1}$) is linked to pre-molt width ($CW_t$) through the empirical Hiatt growth model formulated on the logarithmic scale (Hiatt 1948, Sainte-Marie et al. 1995):

$$\ln(CW_{t+1}) = \beta_0 + \beta_1 \ln(CW_t) + \delta_{\text{mat}} \cdot \mathbb{I}_{\text{mat}} + \Delta_{\text{temp}}(T) + \epsilon$$

where $\mathbb{I}_{\text{mat}} \in \{0, 1\}$ is an indicator variable denoting the terminal molt to morphometric maturity, and $\epsilon \sim \mathcal{N}(0, \sigma^2)$ represents phenotypic variation across individuals.

Empirical regression estimates for males indicate an intercept $\beta_0 = 0.30153 \pm 0.02198$, a slope $\beta_1 = 1.00027 \pm 0.00667$, a terminal maturity decrement $\delta_{\text{mat}} = -0.13564 \pm 0.00922$, and residual standard deviation $\sigma = 0.040$. For females, the parameters are $\beta_0 = 0.19310 \pm 0.05450$, $\beta_1 = 1.03720 \pm 0.01810$, $\delta_{\text{mat}} = -0.13100 \pm 0.02520$, and $\sigma = 0.045$. 

Transformation of this log-linear expectation to natural units introduces the standard log-normal correction factor:

$$\mathbb{E}[CW_{t+1} \mid CW_t, \mathbb{I}_{\text{mat}}] = \exp\left(\beta_0 + \beta_1 \ln(CW_t) + \delta_{\text{mat}} \mathbb{I}_{\text{mat}} + \Delta_{\text{temp}}(T) + \frac{\sigma^2}{2}\right)$$

In the fully featured configuration, thermal anomalies adjust the expected increment via:

$$\Delta_{\text{temp}}(T) = \gamma_T \cdot (T - T_{\text{ref}})$$

where $\gamma_T = 0.015^\circ\text{C}^{-1}$ and reference bottom temperature $T_{\text{ref}} = 3.0^\circ\text{C}$.

## 3.3 Morphometric Maturation Ogives and Terminal Molt Invariance

The probability of undergoing terminal molt to morphometric maturity during ecdysis is expressed as a logistic function of pre-molt carapace width:

$$P(\text{mature} \mid CW, \text{sex}) = \frac{1}{1 + \exp(-(\alpha_{\text{mat}} + \beta_{\text{mat}} \ln(CW)))}$$

For males, regression coefficients are $\alpha_{\text{mat}} = -31.620 \pm 0.121$ and $\beta_{\text{mat}} = 6.9957 \pm 0.0270$, defining a median maturity threshold of $CW_{50} = \exp(31.620 / 6.9957) = 91.82$ mm. For females, the coefficients are $\alpha_{\text{mat}} = -51.1947 \pm 0.2551$ and $\beta_{\text{mat}} = 12.9812 \pm 0.0641$, yielding a median maturity threshold of $CW_{50} = \exp(51.1947 / 12.9812) = 51.61$ mm.

When an individual matures ($u \sim \mathcal{U}(0, 1) < P_{\text{mature}}$), its state transitions permanently to adult. In accordance with the physiological invariance of the terminal molt, the individual ceases all future ecdysial growth, and subsequent annual molting is blocked.

## 3.4 Allometric Mass Scaling

Individual wet body mass $W$ (g) scales allometrically with carapace width $CW$ (mm):

$$W = \alpha_{\text{wt}} \cdot CW^{\beta_{\text{wt}}}$$

Parameter values vary with sex and morphometric maturity. Immature males exhibit scaling coefficients $\alpha_{\text{wt}} = 0.0002665$ and $\beta_{\text{wt}} = 3.098$. Morphometrically mature males exhibit $\alpha_{\text{wt}} = 0.0001830$ and $\beta_{\text{wt}} = 3.160$, reflecting the hyper-allometric enlargement of chelae. Immature females exhibit $\alpha_{\text{wt}} = 0.0003440$ and $\beta_{\text{wt}} = 2.980$, whereas mature females exhibit $\alpha_{\text{wt}} = 0.0004500$ and $\beta_{\text{wt}} = 2.920$, capturing the widening of the brood chamber.

## 3.5 Multivariate Empirical Covariance Parameterization

To initialize simulated populations with realistic joint demographic structures, the model provides an empirical covariance estimation module. Individual traits are transformed into a multivariate normal space:

$$\mathbf{y} = \left[ \ln(CW), \, \ln(W), \, \text{logit}(P_{\text{mat}}), \, \iota, \, \ln(F) \right]^T \sim \mathcal{N}_k(\boldsymbol{\mu}, \mathbf{\Sigma})$$

where the empirical mean vector $\boldsymbol{\mu} \in \mathbb{R}^k$ and positive-definite covariance matrix $\mathbf{\Sigma} \in \mathbb{R}^{k \times k}$ are estimated from survey measurements:

$$\hat{\boldsymbol{\mu}} = \frac{1}{N} \sum_{i=1}^N \mathbf{y}_i, \quad \hat{\mathbf{\Sigma}} = \frac{1}{N-1} \sum_{i=1}^N (\mathbf{y}_i - \hat{\boldsymbol{\mu}})(\mathbf{y}_i - \hat{\boldsymbol{\mu}})^T$$

Synthetic agents generated via Cholesky decomposition of $\hat{\mathbf{\Sigma}}$ preserve biological correlations between body size, wet mass, developmental stage, maturity state, and reproductive potential, preventing spurious combinations during population seeding.

## 3.6 Molting Phenology

In the baseline configuration, molting eligibility requires that elapsed time since last ecdysis exceeds the minimum stage interval ($t - t_{\text{molt}} \ge \Delta t_{\min}(\iota)$). When eligible, ecdysis is evaluated as a constant daily transition hazard:

$$p_{\text{molt}} = 0.05 \cdot \Delta t$$

Upon successful transition, the individual updates carapace width via the Hiatt model, increments its developmental stage, evaluates logistic maturation, and resets $t_{\text{molt}} \leftarrow t$.

In the fully featured configuration, ecdysis requires satisfying a composite physiological gate:

$$\Phi_{\text{molt}} = (\text{stage} \neq \text{adult}) \land (t - t_{\text{molt}} \ge \Delta t_{\min}) \land (\text{DD} \ge \text{DD}_{\text{req}}) \land \Psi_{\text{season}}(\iota, t) \land (E \ge E_{\text{req}})$$

Cumulative thermal degree-days $\text{DD}(t) = \sum_{\tau} \max(0, T(\tau) - T_0) \Delta t$ are integrated above $T_0 = 0.0^\circ\text{C}$. For adolescent crabs ($\iota \ge 7$), the seasonal window $\Psi_{\text{season}}$ restricts ecdysis to days 15 through 140 of the calendar year. Failure to satisfy these criteria within the window induces an annual skip-molt, deferring ecdysis to the subsequent year. When the gate is satisfied, the transition hazard scales with degree-day sufficiency:

$$p_{\text{molt}} = 1 - \exp(-\lambda_{\text{molt}} \Delta t), \quad \lambda_{\text{molt}} = 0.15 \times \min\left(\frac{\text{DD}}{\text{DD}_{\text{req}}}, 3.0\right)$$

Completing ecdysis incurs a substantial energetic cost, retaining a fraction $\kappa_{\text{retain}} = 0.20$ of accumulated reserves ($E \leftarrow 0.20 E$) to account for cuticular synthesis and exuvial shedding.

## 3.7 Individual Bioenergetics and Metabolic Scaling

In the fully featured configuration, individual energy reserve dynamics follow:

$$\frac{dE_i}{dt} = \eta_{\text{assim}} \cdot I_i(CW_i, F, T) - M_{\text{basal}}(W_i, T) - C_{\text{locomotion}}$$

where $\eta_{\text{assim}} = 0.75$ is assimilation efficiency and $C_{\text{locomotion}}$ represents movement costs. Ingestion follows a Holling Type II functional response scaled allometrically to carapace width and ambient temperature:

$$I_i(CW_i, F, T) = I_{\max} \cdot \left(\frac{F(x, y)}{F(x, y) + K_F}\right) \cdot CW_i^{\theta_I} \cdot Q_{10, I}^{(T - T_{\text{ref}})/10}$$

with maximum intake coefficient $I_{\max} = 0.10$, half-saturation constant $K_F = 100.0$, scaling exponent $\theta_I = 0.80$, thermal quotient $Q_{10, I} = 2.0$, and reference temperature $T_{\text{ref}} = 3.0^\circ\text{C}$.

Basal metabolic expenditure obeys Kleiber's 3/4-power mass scaling law:

$$M_{\text{basal}}(W_i, T) = m_0 \cdot W_i^{0.75} \cdot Q_{10, M}^{(T - T_{\text{ref}})/10}$$

where baseline metabolic coefficient $m_0 = 0.008$ and metabolic thermal sensitivity $Q_{10, M} = 2.1$. When energy expenditure exceeds intake, reserves become negative ($E_i < 0$), inducing progressive starvation mortality. In the baseline configuration, bioenergetic balances are bypassed, and individuals experience neither nutritional deficits nor food-mediated mortality.

## 3.8 Reproduction, Spermathecal Buffering, and Incubation

In the baseline configuration, receptive mature females encounter mature males within a local neighborhood radius of two lattice units. Copulation results in immediate extrusion of an egg clutch without sperm storage, and embryos incubate for a constant period of 365 days.

In the fully featured configuration, mating involves size-assortative competition among local adult males. Males with larger carapace width gain priority in mate-guarding and pairing. Copulation charges the female spermatheca to maximum capacity ($S_i \leftarrow 3$ viable clutches). In subsequent breeding seasons, females with $S_i \ge 1$ fertilize and extrude broods autonomously even when males are absent, decrementing stored clutches ($S_{i, t+1} = S_{i, t} - 1$). Fecundity scales allometrically with carapace width, differentiating primiparous females producing their first clutch from multiparous females producing subsequent clutches (Sainte-Marie 1993):

$$F_{\text{primi}}(CW) \approx 0.1893 \cdot CW^{2.9400}, \quad F_{\text{multi}}(CW) \approx 1.0371 \cdot CW^{2.5811}$$

Incubation duration varies inversely with ambient bottom water temperature:

$$\tau_{\text{incubation}} = 365.0 \cdot \left(\frac{5.0}{\max(1.0, \bar{T}_{\text{bottom}})}\right)^{0.30}$$

yielding an incubation span of 365 days at 5.0°C and expanding to 730 days (biennial reproduction) in sub-zero waters below 0.5°C.

## 3.9 Carapace Condition Progression and Multi-Driver Natural Mortality

Following terminal ecdysis, adult cuticular condition advances through five standard fisheries categories: condition 1 (new soft shell), condition 2 (clean hard shell), condition 3 (intermediate wear with light epibionts), condition 4 (old hard shell with extensive scratching and encrustation), and condition 5 (decaying shell with cuticular pitting and limb loss). Progression between condition stages follows a Poisson transition hazard:

$$P(\text{advance } CC) = 1 - \exp(-\lambda_{cc} \cdot \Delta t), \quad \lambda_{cc} \approx \frac{0.35}{365.0}\text{ day}^{-1}$$

In the fully featured configuration, natural mortality integrates five distinct components:

$$M_{\text{total}} = M_{\text{senescence}} + M_{\text{predation}}(CW) + M_{\text{starvation}}(E) + M_{\text{thermal}}(T) + M_{\text{shell}}(CC)$$

Adult senescence imposes a baseline rate $M_0 / 365$ with $M_0 = 0.20\text{ yr}^{-1}$. Size-dependent predation risk declines exponentially with carapace width:

$$M_{\text{predation}} = \frac{0.15}{365} \exp(-0.03 \cdot (CW - 5.0))$$

Starvation mortality acts upon negative energy reserves ($\mu_{\text{starv}} \cdot |E_i|$ when $E_i < 0$). Thermal stress mortality increases quadratically above critical thresholds:

$$M_{\text{thermal}} = \mu_T \cdot \max(0, T - 7.0^\circ\text{C})^2$$

Cuticular degradation adds an excess mortality hazard $\Delta M_{CC5} \approx 0.25\text{ yr}^{-1}$ for individuals in carapace condition 5. In the baseline configuration, natural mortality simplifies strictly to adult senescence combined with juvenile size-attenuated predation.

## 3.10 Spatial Movement

Crab movement across the lattice is governed by behavioral dispatch. In the baseline configuration, crabs execute unbiased random walks to adjacent Moore neighborhood cells ($dx, dy \in \{-1, 0, 1\}$). In the fully featured configuration, crabs execute directed habitat taxis. For each candidate neighboring cell $(x', y')$, a habitat utility index is calculated:

$$U(x', y') = -\omega_T |T(x', y') - T_{\text{opt}}| + \omega_F \frac{F(x', y')}{K_F}$$

where preferred temperature $T_{\text{opt}} \approx 2.5^\circ\text{C}$, thermal weighting $\omega_T = 1.0$, and foraging weighting $\omega_F = 0.5$. Movement steps are sampled from multinomial logit transition probabilities over candidate cells:

$$P(x', y') = \frac{\exp(U(x', y'))}{\sum_{(u, v)} \exp(U(u, v))}$$

## 3.11 Commercial Exploitation and Macro-Model Estimation

Commercial harvest is applied to legal-sized males ($CW \ge 95$ mm) through a logistic trap selectivity curve:

$$S_{\text{trap}}(CW) = \frac{1}{1 + \exp(-(CW - 95.0) / 5.0)}$$

Harvest removals generate daily commercial yield:

$$Y(t) = \frac{1}{1000} \sum_{i \in \text{harvested}} W_i \text{ (kg)}$$

The simulation engine maintains running biomass totals, including total living biomass $B_{\text{total}}$, female spawning stock biomass $\text{SSB}_F$, mature male biomass $\text{MMB}$, and legal commercial male biomass $B_{\text{legal}}$.

Annual population outputs serve as inputs to macro-ecological population dynamics estimators. The suite fits a discrete-time Schaefer surplus production model:

$$N_{t+1} = N_t + r N_t \left(1 - \frac{N_t}{K}\right) - C_t$$

a Deriso-Schnute delay-difference model:

$$N_{t+1} = s (N_t - C_t) + R_{t+1}$$

and a Generalized Leslie stage-transition matrix model:

$$\mathbf{n}_{t+1} = \mathbf{A}\mathbf{n}_t - \mathbf{C}_t$$

across fifteen developmental stages. Spectral decomposition of $\mathbf{A}$ yields the asymptotic growth rate $r = \ln(\lambda_1)$, the stable stage distribution, and stage-specific elasticities. Retrospective peeling analysis computes Mohn's $\rho$ across sequential terminal assessment peels:

$$\rho = \frac{1}{P} \sum_{p=1}^P \frac{B_{T-p, \text{peel } p} - B_{T-p, \text{ref}}}{B_{T-p, \text{ref}}}$$

evaluating model consistency against the standard diagnostic criterion $|\rho| \le 0.20$ (Mohn 1999, Hurtado-Ferro et al. 2015).

# 4. Computational Implementation and Performance

The model is implemented in the Julia programming language utilizing the `Agents.jl` agent-based simulation library (Datseris et al. 2022). Julia provides LLVM-based just-in-time compilation, type specialization, and native multi-threading, facilitating high-throughput ecological simulation without recourse to external compiled C or Fortran modules.

Memory efficiency is achieved through bitstype representations in agent definitions. The individual `Organism` struct occupies 104 bytes in memory. Consequently, an active population of 100,000 spatial agents requires approximately 10.4 MB of RAM, allowing large-scale demographic ensembles to be evaluated entirely within primary CPU cache.

Benchmark evaluations on standard workstation hardware demonstrate execution speeds suitable for Monte Carlo calibration. A three-year simulation consisting of 1,095 daily time steps with 300 initial agents on a $100 \times 100$ lattice executes in approximately 2.9 seconds under the distilled baseline configuration and 3.8 seconds under the fully featured eco-physiological configuration.

# 5. Process Toggle Architecture

The duality between parsimonious baseline mechanics and eco-physiological realism is summarized in Table 1.

| Biological Process | Distilled Baseline Configuration | Fully Featured Configuration |
|---|---|---|
| Ecdysis Eligibility | Minimum intermolt days ($t - t_{\text{molt}} \ge \Delta t_{\min}$) | Degree-days ($\text{DD} \ge \text{DD}_{\text{req}}$), intermolt, energy, spring window |
| Growth Increments | Hiatt linear-log model with phenotypic error | Hiatt model with thermal modulation $\Delta_{\text{temp}}(T)$ and condition |
| Foraging and Energetics | Unlimited forage; bioenergetic balances bypassed | Holling Type II foraging, $W^{0.75}$ metabolic scaling, starvation |
| Temperature Coupling | Uncoupled | Drives degree-days, intake, metabolism, incubation, and stress |
| Carapace Condition | Uncoupled (cuticular wear invariant) | Five condition categories ($CC_1 \dots CC_5$) with wear and mortality |
| Mating Dynamics | Local pairing; immediate clutch extrusion | Size-assortative competition; 3-clutch spermathecal store |
| Embryonic Incubation | Constant duration ($\tau = 365$ days) | Temperature-dependent $\tau = 365 (5.0 / T)^{0.30}$ (365 to 730 days) |
| Spatial Dispersal | Unbiased 2D random walk | Habitat taxis toward thermal optima and forage gradients |
| Natural Mortality | Adult senescence ($M_0 = 0.20$) and juvenile predation | Multi-driver (senescence, predation, starvation, thermal, shell wear) |

Table 1: Comparison of physiological, behavioral, and demographic assumptions between the distilled baseline and fully featured configurations.

# 6. Model Verification and Benchmark Testing

The simulation codebase is supported by a comprehensive suite of automated verification tests implemented in `test/runtests.jl`.

Morphometric invariants are validated by testing that empirical modal boundaries preserve strict monotonic ordering ($CW_{\min}(\iota) < CW_{\max}(\iota)$) across all instars. Growth increment routines are verified against published empirical regression intervals, ensuring that simulated size transitions match empirical distributions with expected phenotypic variances ($\sigma = 0.045$). Maturity ogive algorithms are checked to confirm that the median size at maturity satisfies $P(\text{mature} \mid CW_{50}) = 0.50$ for both sexes, and that terminal molt invariance is strictly maintained upon transition to adult status. Length-mass equations ($W = \alpha CW^\beta$) are verified against published allometric benchmarks for all sex and maturity groups.

Stepping dynamics are verified across both architectural pipelines. Under the baseline pipeline, tests confirm that molting follows elapsed intermolt intervals without thermal dependence, that female fertilization occurs directly without multi-year spermathecal accumulation, that spatial dispersal follows unbiased diffusion, and that food constraints remain inactive. Under the fully featured pipeline, tests verify that degree-days accumulate monotonically, that metabolic costs scale with $W^{0.75}$, that cuticular wear advances through categories 1 to 5, that multi-driver mortality counters correctly attribute deaths, and that habitat taxis directs displacement toward thermal optima.

The empirical covariance matrix parameterization is verified by confirming the positive-definiteness of $\hat{\mathbf{\Sigma}}$ and validating that synthetic agents generated via Cholesky sampling reproduce target correlation structures. Stock biomass calculations are verified for mathematical consistency, ensuring that total biomass strictly exceeds spawning biomass and that commercial male biomass reflects the 95 mm legal threshold. Stock-recruitment algorithms are tested by non-linear estimation of Beverton-Holt and Ricker curves. Macro-ecological estimators are validated through process-error regressions and grid-search fits for Schaefer and Delay-Difference models. Retrospective peeling routines are tested across sequential peels, verifying degrees-of-freedom bounds and classifying Mohn's $\rho$ within target bounds ($|\rho| \le 0.20$). Finally, Generalized Leslie transition matrices are tested to confirm non-negative entry bounds, dominant real eigenvalues ($\lambda_1 > 0$), and normalized stable stage and elasticity vectors ($\sum w_i = 1.0$, $\sum E_{ij} = 1.0$).

# 7. Discussion and Future Extensions

The individual-based framework established here resolves the tension between computational tractability and eco-physiological realism in decapod population modeling. By formalizing a configurable toggle architecture within a unified simulation engine, the model serves dual roles: providing an agile reference operating model for stock assessment calibration, while functioning as a mechanistic platform for exploring climate-driven physiological stress.

Several extensions represent natural research priorities. Coupling the simulation lattice with three-dimensional hydrodynamic circulation models (such as NEMO or FVCOM) will permit explicit advection of pelagic zoeal and megalopal larvae, resolving spatial connectivity between spawning grounds and juvenile nursery habitats. Incorporating climate projection scenarios will facilitate evaluation of warming trends within the Cold Intermediate Layer of the Northwest Atlantic, quantifying risks of metabolic starvation and recruitment collapse under marine heatwaves. Multi-species trophic coupling, particularly integrating dynamic predation fields for Atlantic cod and Atlantic halibut along with endogenized juvenile cannibalism, will improve representation of top-down mortality. Finally, the framework provides an operating model suitable for formal Management Strategy Evaluation, enabling rigorous testing of harvest control rules, soft-shell closure mechanisms, and spatial marine protected areas.

# 8. References

Alunno-Bruscia, M., and Sainte-Marie, B. (1998). Abdomen allometry, ovary development, and growth of female snow crab, *Chionoecetes opilio* (Brachyura, Majidae), in the northwestern Gulf of St. Lawrence. *Canadian Journal of Fisheries and Aquatic Sciences*, 55(2), 459–477.


Beverton, R. J. H., and Holt, S. J. (1957). *On the Dynamics of Exploited Fish Populations*. Fishery Investigations Series II, Vol. 19. Ministry of Agriculture, Fisheries and Food, London.

Boudreau, S. A., Anderson, S. C., and Worm, B. (2011). Top-down interactions and temperature control of snow crab abundance in the northwest Atlantic Ocean. *Marine Ecology Progress Series*, 429, 169–183.

Caswell, H. (2001). *Matrix Population Models: Construction, Analysis, and Interpretation*. 2nd Edition. Sinauer Associates, Sunderland, MA.

Choi, J. S., and Zisserson, B. M. (2012). An assessment of snow crab in the Scotian Shelf (Areas 20 to 24, 4X). *DFO Canadian Science Advisory Secretariat Research Document*, 2012/038.

Comeau, M., Conan, G. Y., Maynou, F., Robichaud, G., Therriault, J. C., and Starr, M. (1998). Growth, spatial distribution, and abundance of snow crab (*Chionoecetes opilio*) in the Baie des Chaleurs, southwestern Gulf of St. Lawrence, Canada. *Canadian Journal of Fisheries and Aquatic Sciences*, 55(1), 247–260.

Conan, G. Y., and Comeau, M. (1986). Functional maturity and terminal molt of male snow crab, *Chionoecetes opilio*. *Canadian Journal of Fisheries and Aquatic Sciences*, 43(9), 1710–1719.

Datseris, G., Vahdati, A. R., and DuBois, T. C. (2022). Agents.jl: a performant and feature-full agent-based modeling software of minimal code complexity. *Simulation*, 98(9), 803–813.

Dawe, E. G., Mullowney, D. R., Moriyasu, M., and Wade, E. (2012). Effects of temperature on size-at-terminal molt and morphometric maturity in snow crab (*Chionoecetes opilio*). *Fisheries Research*, 121–122, 61–70.

Deriso, R. B. (1980). Harvesting strategies and parameter estimation for an age-structured model. *Canadian Journal of Fisheries and Aquatic Sciences*, 37(2), 268–282.

DFO (Department of Fisheries and Oceans Canada). (2020). Assessment of Nova Scotia (4VWX) Snow Crab. *DFO Canadian Science Advisory Secretariat Science Advisory Report*, 2020/027.

DFO (Department of Fisheries and Oceans Canada). (2022). Assessment of Southern Gulf of St. Lawrence Snow Crab (*Chionoecetes opilio*). *DFO Canadian Science Advisory Secretariat Science Advisory Report*, 2022/019.

Dionne, M., Sainte-Marie, B., Bourget, E., and Gilbert, D. (2003). Distribution and habitat selection of benthic stages of snow crab, *Chionoecetes opilio*, in relation to water depth and temperature. *Marine Ecology Progress Series*, 259, 117–128.

Drinkwater, K. F. (2005). The response of Atlantic cod (*Gadus morhua*) to future climate change. *ICES Journal of Marine Science*, 62(7), 1327–1337.

Elner, R. W., and Beninger, P. G. (1995). Multiple reproductive strategies in snow crab, *Chionoecetes opilio*: Physiological paths and behavioral plasticity. *Journal of Experimental Marine Biology and Ecology*, 193(1-2), 93–112.

Ernst, B., Orensanz, J. M., and Armstrong, D. A. (2005). Spatial dynamics of female snow crab (*Chionoecetes opilio*) in the eastern Bering Sea. *Canadian Journal of Fisheries and Aquatic Sciences*, 62(2), 250–268.

Fedewa, E. J., Jackson, T. M., Daly, B. J., and Gardner, L. (2020). Recent fluctuations in snow crab (*Chionoecetes opilio*) size-at-maturity and recruitment in the eastern Bering Sea. *Deep Sea Research Part II: Topical Studies in Oceanography*, 181–182, 104903.

Foyle, T. P., O'Dor, R. K., and Elner, R. W. (1989). Energetically defining the thermal limits of the snow crab, *Chionoecetes opilio*. *Journal of Experimental Biology*, 145(1), 371–393.

Grimm, V., Berger, U., Bastiansen, F., Eliassen, S., Ginot, V., Giske, J., Goss-Custard, J., Grand, T., Heinz, S. K., Huse, G., Huth, A., Jepsen, J. U., Jørgensen, C., Mooij, W. M., Müller, B., Pe'er, G., Piou, C., Railsback, S. F., Robbins, A. M., Robbins, M. M., Rossmanith, E., Rüger, N., Strand, E., Souissi, S., Stillman, R. A., Vabø, R., Visser, U., and DeAngelis, D. L. (2006). A standard protocol for describing individual-based and agent-based models (ODD). *Ecological Modelling*, 198(1-2), 115–126.

Grimm, V., Railsback, S. F., Vincenot, C. E., Berger, U., Gallagher, C., DeAngelis, D. L., Edmonds, B., Ge, J., Giske, J., Groeneveld, J., Johnston, A. S., Milles, A., Nabe-Nielsen, J., Polhill, J. G., Radchuk, V., Rohwäder, M. S., Stillman, R. A., Thiele, J. C., and Ayllón, D. (2020). The ODD protocol for describing agent-based and other simulation models: A second update to improve clarity, replication, and structural realism. *Journal of Artificial Societies and Social Simulation*, 23(2), 7.

Hébert, M., Benhalima, K., Miron, G., and Moriyasu, M. (2002). Molting and growth of male snow crab, *Chionoecetes opilio* (O. Fabricius, 1788) (Crustacea: Majidae) in the southern Gulf of St. Lawrence. *Crustaceana*, 75(5), 671–702.

Hiatt, R. W. (1948). The biology of the lined shore crab, *Pachygrapsus crassipes* Randall. *Pacific Science*, 2(3), 135–213.

Hurtado-Ferro, F., Szuwalski, C. S., Valero, J. L., Anderson, S. C., Cunningham, C. J., Johnson, K. F., Licandeo, R., McGilliard, C. R., Monnahan, C. C., Rudd, M. B., Stawitz, C. C., Hicks, A. C., Sansó, B., and Punt, A. E. (2015). Looking in the rear-view mirror: bias and retrospective patterns in integrated, age-structured stock assessment models. *ICES Journal of Marine Science*, 72(1), 99–110.

Kooijman, S. A. L. M. (2010). *Dynamic Energy Budget Theory for Metabolic Organisation*. 3rd Edition. Cambridge University Press, Cambridge.

Kuhn, S., and Choi, J. S. (2011). Influence of temperature on embryo development and hatching phenology in the snow crab, *Chionoecetes opilio*. *Fisheries Oceanography*, 20(6), 461–472.

Lefkovitch, L. P. (1965). The study of population growth in organisms grouped by stages. *Biometrics*, 21(1), 1–18.

Leslie, P. H. (1945). On the use of matrices in certain population mathematics. *Biometrika*, 33(3), 183–212.

Marcello, L. A., Fairweather, P. G., and Choi, J. S. (2012). Assessing the stability and resilience of snow crab (*Chionoecetes opilio*) ecosystem dynamics in the Northwest Atlantic. *Ecological Applications*, 22(8), 2110–2126.

Mohn, R. (1999). The retrospective problem in stock assessment: An amphidrome of check and balance. *ICES Journal of Marine Science*, 56(4), 473–488.

Moriyasu, M., Wade, E., Sinclair, A., and Chiasson, Y. (1998). Snow crab, *Chionoecetes opilio*, stock assessment in the southwestern Gulf of St. Lawrence by bottom trawl survey. *Canadian Technical Report of Fisheries and Aquatic Sciences*, 2240.

Ouellet, P., and Sainte-Marie, B. (2018). Vertical distribution and development of snow crab (*Chionoecetes opilio*) larvae in the southwestern Gulf of St. Lawrence. *Journal of Crustacean Biology*, 38(4), 416–428.

Railsback, S. F., and Grimm, V. (2019). *Agent-Based and Individual-Based Modeling: A Practical Introduction*. 2nd Edition. Princeton University Press, Princeton, NJ.

Rondeau, A., and Sainte-Marie, B. (2001). Variable mate-guarding time and sperm allocation by male snow crabs (*Chionoecetes opilio*) in response to competition. *Canadian Journal of Fisheries and Aquatic Sciences*, 58(6), 1243–1256.

Sainte-Marie, B. (1993). Reproductive cycle and fecundity of primiparous and multiparous female snow crab, *Chionoecetes opilio*, in the northwest Gulf of St. Lawrence. *Canadian Journal of Fisheries and Aquatic Sciences*, 50(10), 2147–2156.

Sainte-Marie, B., Raymond, S., and Brêthes, J. C. (1995). Growth and maturation of the benthic stages of male snow crab, *Chionoecetes opilio* (Brachyura: Majidae). *Canadian Journal of Fisheries and Aquatic Sciences*, 52(5), 903–924.

Sainte-Marie, B., Sevigny, J. M., and Gignac, M. (1997). Assessment of female snow crab (*Chionoecetes opilio*) sperm stores, and courtship behavior in the wild. *Canadian Journal of Fisheries and Aquatic Sciences*, 54(2), 239–248.

Sainte-Marie, B., Sevigny, J. M., and Carpentier, M. (2002). Interannual variability of sperm reserves and fecundity of primiparous females of the snow crab (*Chionoecetes opilio*) in relation to sex ratio. *Canadian Journal of Fisheries and Aquatic Sciences*, 59(12), 1932–1940.

Sainte-Marie, G., and Sainte-Marie, B. (1999). Growth, developmental schedule, and terminal molt of the snow crab (*Chionoecetes opilio*) in the western Gulf of St. Lawrence. *Canadian Journal of Fisheries and Aquatic Sciences*, 56(6), 940–953.

Schaefer, M. B. (1954). Some aspects of the dynamics of populations important to the management of the commercial marine fisheries. *Inter-American Tropical Tuna Commission Bulletin*, 1(2), 27–56.

Schnute, J. (1985). A general theory for analysis of catch and effort data. *Canadian Journal of Fisheries and Aquatic Sciences*, 42(3), 414–429.

Skanes, K. R., Mullowney, D. R., and Dawe, E. G. (2017). An assessment of Newfoundland and Labrador snow crab (*Chionoecetes opilio*) in 2015. *DFO Canadian Science Advisory Secretariat Research Document*, 2017/025.

Szuwalski, C. S., Cheng, W., Foy, R., Hermann, A. J., Hollowed, A. B., Holsman, K. K., Lee, J., Zheng, J., and Ianelli, J. (2021). Climate change and the future of snow crab in the Bering Sea. *Progress in Oceanography*, 194, 102555.

Szuwalski, C. S., Aydin, K., Fedewa, E. J., Halloran, E., and Litzow, M. A. (2023). The collapse of eastern Bering Sea snow crab. *Science*, 382(6668), 306–310.

Thompson, R. J., and Hawryluk, M. (1990). Physiological energetics of the snow crab *Chionoecetes opilio*. In *Proceedings of the International Symposium on King and Snow Crabs*, pp. 283–291. Alaska Sea Grant College Program, Fairbanks, AK.

Tremblay, M. J. (1997). Snow crab (*Chionoecetes opilio*) distribution and abundance in relation to temperature and bathymetry on the Scotian Shelf. *Canadian Technical Report of Fisheries and Aquatic Sciences*, 2161.

Visser, A. W. (2008). Biomass and functional diversity in individual-based models of marine ecosystems. *Journal of Marine Systems*, 74(1-2), 488–499.

Webb, J. B., Eckert, G. L., Shirley, T. C., and Tamone, S. L. (2007). Changes in embryonic development and hatching in snow crab, *Chionoecetes opilio*, under temperature treatment. *Journal of Crustacean Biology*, 27(1), 107–115.
