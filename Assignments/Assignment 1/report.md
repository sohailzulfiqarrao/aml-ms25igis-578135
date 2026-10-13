# Applied Machine Learning Assignment 1 – Predicting Evapotranspiration over Islamabad: Report

| | |
|---|---|
| **Name** | Sohail Zulfiqar |
| **Registration ID** | 578135 |
| **Email** | szulfiqar.ms25igis@student.nust.edu.pk |
| **Program** | MS GIS & RS – 2025 |
| **Subject** | Applied Machine Learning (Hammad Gilani) |
| **Repository** | https://github.com/sohailzulfiqarrao/aml-ms25igis-578135 (folder `Assignments/Assignment 1`) |
| **Live page** | https://sohailzulfiqarrao.github.io/aml-ms25igis-578135/Assignments/Assignment%201/ |
| **Notebook** | [`Islamabd_MLR_Assignment.ipynb`](Islamabd_MLR_Assignment.ipynb), data in [`Islamabad_Data.csv`](Islamabad_Data.csv) |

## 1. Objective and data

The aim is to predict surface evapotranspiration (`ET_mm`) across the Islamabad Capital Territory from satellite-derived vegetation indices and climate variables, using multiple linear regression. `Islamabad_Data.csv` has **87 observations** and eight candidate predictors:

| Variable | Meaning |
|---|---|
| `NDVI`, `EVI` | Normalized Difference and Enhanced Vegetation Indices |
| `LAI` | Leaf Area Index |
| `LST_Day_C` | Daytime land surface temperature, °C |
| `T2m_C` | Air temperature at 2 m, °C |
| `Prec_mm` | Precipitation, mm |
| `SoilM_L1_vfrac` | Soil moisture, layer 1, volume fraction |
| `Wind10m_ms` | Wind speed at 10 m, m/s |

## 2. Method

1. **Completeness audit.** All 87 rows are complete: 0 nulls and no sentinel values.
2. **Outliers.** IQR fences (Q1 − 1.5 IQR, Q3 + 1.5 IQR) flag LAI (10 values), wind (5), daytime LST (2), precipitation (1) and 2 m temperature (1). Each value is capped at its fence, so no row is dropped.
3. **Correlation and VIF.** NDVI, EVI and LAI are strongly correlated with each other (EVI–NDVI r = 0.95). EVI has a VIF of 17.90 and is removed. NDVI then falls from 29.98 to 9.17 and LAI from 11.35 to 11.06.
4. **Split and scaling.** An 80/20 split with seed 42 gives 69 training rows and 18 test rows. The seven remaining predictors are z-score scaled with a scaler fitted on the training rows only.
5. **Fit and evaluate.** Ordinary least squares, scored on both partitions and with 5-fold shuffled cross-validation (seed 42).
6. **Inference.** The model is refitted with statsmodels on the 69 training rows for coefficients, p-values and residual tests.

## 3. Results

### Train and test scores

| Partition | MAE | MSE | RMSE | R² | Adj. R² | MAPE % |
|---|---|---|---|---|---|---|
| Training (69 rows) | 1.0052 | 2.2436 | 1.4979 | 0.8147 | 0.7934 | 10.99 |
| **Testing (18 rows)** | 0.9740 | 3.1649 | 1.7790 | **0.7850** | 0.6345 | 9.07 |

### 5-fold cross-validation

| Fold | R² | RMSE (mm) |
|---|---|---|
| 1 | 0.7850 | 1.7790 |
| 2 | 0.5754 | 1.9830 |
| 3 | 0.6963 | 1.9620 |
| 4 | 0.8124 | 1.7098 |
| 5 | 0.6498 | 1.7678 |
| **Mean (± std)** | **0.7038 (±0.0870)** | **1.8403 (±0.1106)** |

### Fitted equation

```
ET_mm = 12.8571 - 0.0712(LAI) - 0.1171(LST_Day_C) + 16.8534(NDVI) - 0.0012(Prec_mm)
        + 14.2934(SoilM_L1_vfrac) - 0.9464(T2m_C) + 6.7701(Wind10m_ms)
```

| Predictor | Coefficient | Standardized β | p-value | VIF |
|---|---|---|---|---|
| **NDVI** | 16.8534 | 2.4528 | < 0.001 | 9.17 |
| **SoilM_L1_vfrac** | 14.2934 | 0.6685 | 0.034 | 2.52 |
| T2m_C | -0.9464 | -0.6042 | 0.155 | 4.74 |
| Wind10m_ms | 6.7701 | 0.2130 | 0.336 | 1.41 |
| Prec_mm | -0.0012 | -0.1806 | 0.658 | 4.52 |
| LST_Day_C | -0.1171 | -0.1690 | 0.712 | 3.99 |
| LAI | -0.0712 | -0.0316 | 0.964 | 11.06 |

NDVI is by far the strongest predictor: its standardized β of 2.45 is more than three times the next largest in size. Only NDVI and soil moisture are significant at the 5% level.

## 4. Figures

### Correlation matrix
![Correlation matrix of evapotranspiration and the eight predictors](images/correlation-matrix.png)

NDVI, EVI and LAI correlate with ET at 0.89, 0.85 and 0.87, and with each other at 0.88 to 0.95. This is the collinearity the VIF step addresses.

### Variance inflation factors
![VIF before and after pruning EVI](images/vif-comparison.png)

Removing EVI brings NDVI under the threshold of 10 (30.0 to 9.2). LAI stays just above it (11.3 to 11.1).

### Observed against predicted, and residual distribution
![Observed against predicted ET and the residual histogram](images/model-evaluation.png)

Most of the 18 test points sit close to the 1:1 line. The highest observed values (about 14 mm and above) are predicted too low, which gives the long right tail in the residual histogram.

### Residuals against fitted values
![Residuals against fitted values on the test set](images/residuals-vs-fitted.png)

For predicted values below 11 mm the residuals stay within about ±0.8 mm. Above that the spread widens and the largest residuals appear, including one of about 6.5 mm and one of about 3.1 mm.

## 5. Diagnostics and limits

- **Collinearity is reduced, not removed.** LAI is still at VIF 11.06, above the threshold of 10 that the notebook uses.
- **Residuals are not normal.** In the statsmodels fit on the training rows the Jarque–Bera test gives p = 0.0035 and the omnibus test p = 0.024. The Durbin–Watson statistic is 2.10 (no sign of autocorrelation) and the condition number is 9.38.
- **The test set is small.** With 18 rows the test R² of 0.785 sits within a fold range of 0.58 to 0.81. Fold 1 reproduces the test-set scores exactly (R² 0.7850, RMSE 1.7790 mm), so the cross-validated mean of 0.704 is the steadier figure.
- **Outliers were capped, not removed.** Capping keeps all 87 rows but changes the flagged predictor values, mainly LAI and wind.

## 6. Files and how to run

| File | What it is |
|---|---|
| `Islamabd_MLR_Assignment.ipynb` | The notebook: audit, capping, VIF pruning, scaling, regression and cross-validation |
| `Islamabad_Data.csv` | Its input data |
| `index.html`, `images/` | The results page and the four figures taken from the notebook |

The notebook reads `Islamabad_Data.csv` from its working folder. Run it from this folder with Jupyter, or open it in [Google Colab](https://colab.research.google.com/github/sohailzulfiqarrao/aml-ms25igis-578135/blob/main/Assignments/Assignment%201/Islamabd_MLR_Assignment.ipynb) and upload the CSV first:

```
pip install notebook pandas scikit-learn matplotlib seaborn statsmodels tabulate
jupyter notebook
```

When run, the notebook also writes four figures and five CSV tables (audit, metrics, coefficients, VIF and test predictions) next to itself; they are not committed here.
