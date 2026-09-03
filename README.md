# 💧 Gray-Box ML Framework for Groundwater-Irrigation Response

<!-[License](LICENSE)>
<!-[Paper](https://doi.org/10.3390/w18141661)>
<!-[Python](https://www.python.org/)>

Official code implementation for the paper: **"Gray-Box Machine Learning Framework for Extracting Groundwater–Irrigation Response Functions and Inverting Hydrogeological Parameters"**.

This project proposes a **Gray-Box Machine Learning (ML) framework** to address challenges in groundwater management within over-exploited aquifer systems. Unlike traditional black-box models that directly predict groundwater levels, our framework learns the coefficients of irrigation-groundwater response functions, enabling the inversion of key hydrogeological parameters while maintaining both physical interpretability and computational efficiency.

<img width="1101" height="1371" alt="地下水技术路线图-第 1 页 drawio" src="https://github.com/user-attachments/assets/cd0c1b7a-82fb-4f59-a690-37004fcf6ed0" />

---

## 📖 Introduction

Groundwater irrigation underpins global food security but has also led to severe aquifer depletion. Decision-makers urgently need quantitative tools to answer: "Under specific climate, soil, and cropping regimes, how does irrigation intensity impact the groundwater budget?"

By integrating a **process-based model (SWAT-GW)** with **ensemble machine learning algorithms**, this project establishes a new paradigm that shifts from predicting state variables to extracting functional structures.

### ✨ Key Features
- **Response Function Extraction**: Learns the quadratic polynomial relationships between irrigation intensity and groundwater responses (recharge, storage change, and water table change).
- **Parameter Inversion**: Directly inverts four key management parameters from the ML-predicted coefficients:
  - Precipitation infiltration coefficient ($\alpha$)
  - Irrigation infiltration coefficient ($\beta$)
  - Natural recharge under zero irrigation ($R_{nat}$)
  - Recharge-irrigation equilibrium point ($IRR_{eq}$)
- **Data Scarcity Robustness Testing**: Evaluates model performance under different data availability tiers (Tier 1/2/3).
- **Interpretability Analysis**: Integrates SHAP and Causal Forest analyses to uncover the physical mechanisms driving groundwater responses.

---

## 🛠️ Methodology

The workflow of the framework is as follows:

1. **Data Generation**: Uses a validated **SWAT-GW** model (Piedmont Plain of the North China Plain) to generate training data.
2. **Function Fitting**: Constructs gradient irrigation scenarios to fit quadratic polynomials between irrigation amounts and groundwater response variables.
3. **ML Training**: Uses ensemble algorithms (XGBoost, LightGBM, GBR, RF) to predict the polynomial coefficients.
4. **Parameter Inversion**: Calculates hydrogeological parameters based on the geometric properties of the coefficients.

### Supported Algorithms
- Random Forest (RF)
- Gradient Boosting Regression (GBR)
- XGBoost
- LightGBM

---

## 📊 Data Description

The training data for this project is based on 70 Hydrologic Response Units (HRUs) in the Piedmont Plain of the North China Plain.

### Input Features (26 variables)
Model input features are divided into three tiers to simulate different data scarcity scenarios:

| Tier | Data Type | Example Variables |
| :--- | :--- | :--- |
| **Tier 1** | **Easily Accessible** | Meteorological forcing (precipitation, temperature, radiation), basic soil properties (texture, bulk density, organic carbon) |
| **Tier 2** | **Moderately Accessible** | Soil hydraulic parameters (saturated hydraulic conductivity, field capacity), agricultural management (summer maize irrigation amount) |
| **Tier 3** | **Difficult to Access** | Deep hydrogeological parameters (specific yield `GW_SPYLD`, delay time `GW_DELAY`, lateral flow contribution `LARCHRG`) |

### Output Targets
- **Recharge**: Vertical recharge ($mm/yr$)
- **Storage**: Shallow aquifer storage change ($mm/yr$)
- **Water Table**: Change in shallow groundwater table depth ($m/yr$)

---

## 🚀 Quick Start

### 1. Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install numpy pandas scikit-learn xgboost lightgbm shap grf
