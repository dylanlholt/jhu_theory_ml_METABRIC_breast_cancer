# METABRIC Breast Cancer Survival Analysis

## Survival Analysis and Predictive Modeling Using Machine Learning

**Authors:** Miro Everaert and Dylan Holt  
**Date:** August 10, 2025  
**Course:** Graduate Mathematics - Final Project

---

## Abstract

This study leveraged the METABRIC dataset (n=1,904) to compare traditional statistical methods with modern machine learning approaches for breast cancer survival prediction. We analyzed 31 clinical attributes and 506 genomic variables, including mRNA expression levels and gene mutations, to predict overall survival outcomes. Principal component analysis was applied to reduce dimensionality of the 489 gene mutation features, with the first 25 components capturing ~40% of explained variance.

Multiple linear regression for survival time prediction showed modest improvement when incorporating genomic data (R² increase from 0.15 to 0.25). However, proper survival analysis revealed more dramatic differences between methodological approaches. Cox Proportional Hazards regression with clinical features achieved a concordance index of 0.626, while Random Survival Forest achieved 0.630. When genomic features were incorporated, Cox regression performance degraded (C-index: 0.578), whereas Random Survival Forest showed substantial improvement (C-index: 0.718).

The results demonstrate that non-linear machine learning methods are essential for leveraging high-dimensional genomic data in survival prediction. The 14% improvement in discriminative ability (C-index: 0.626 → 0.718) represents clinically meaningful enhancement in patient risk stratification. This work illustrates the transition from traditional clinical staging toward molecular-based prognostication, advancing precision medicine approaches in oncology.

---

## Project Overview

### Dataset
- **Source:** METABRIC (Molecular Taxonomy of Breast Cancer International Consortium)
- **Sample Size:** 1,904 primary breast cancer patients
- **Study Period:** 2006-2012 with up to 20+ years follow-up
- **Geographic Scope:** UK and Canada

### Features
- **Clinical Variables (31):** Demographics, tumor characteristics, treatment history, pathology markers
- **Genomic Variables (506):** mRNA expression z-scores (331 genes) + mutation status (175 genes)
- **Survival Endpoints:** Overall Survival (OS), Disease-Specific Survival (DSS), Relapse-Free Survival (RFS)

### Methodology

#### 1. Exploratory Data Analysis
- Clinical indicator distributions and epidemiological validation
- Treatment exposure patterns
- Genomic variable distributions and survival relationships

#### 2. Feature Reduction
- Principal Component Analysis on 489 gene mutation features
- Dimensionality reduction to 25 components (~40% explained variance)
- Survival pattern analysis for key genes (TP53, BRCA1, etc.)

#### 3. Predictive Modeling
- **Traditional Approaches:**
  - Multiple Linear Regression for survival time prediction
  - Logistic Regression for death classification
  - Cox Proportional Hazards regression
  
- **Machine Learning Approaches:**
  - Random Survival Forest (RSF)
  - Parameters: n_estimators=200, max_depth=5, min_samples_leaf=20

### Key Results

#### Performance Metrics (Concordance Index)
| Model Design | Clinical Features Only | Plus mRNA Features |
|--------------|----------------------|-------------------|
| Cox Proportional Hazards (Full Dataset) | 0.608 ± 0.020 | 0.606 ± 0.029 |
| Random Survival Forest (Full Dataset) | 0.630 ± 0.025 | **0.701 ± 0.018** |
| Cox Proportional Hazards (Reduced Dataset) | 0.600 ± 0.028 | 0.578 ± 0.025 |
| Random Survival Forest (Reduced Dataset) | 0.626 ± 0.020 | **0.718 ± 0.016** |

#### Key Findings
1. **RSF consistently outperforms Cox regression** in all scenarios
2. **Genomic features provide substantial benefit** when using appropriate ML methods
3. **Traditional Cox regression degrades** with high-dimensional genomic data
4. **14% improvement in C-index** (0.626 → 0.718) represents clinically meaningful enhancement

---

## Clinical Implications

### 1. **Clinical Impact**
Modern ML methods (RSF) are essential for leveraging genomic data in survival prediction, while traditional Cox models remain adequate for clinical-only assessments.

### 2. **Methodological Insight**
Non-linear relationships between genomic features and survival outcomes require sophisticated modeling approaches that can capture complex gene-gene and gene-clinical interactions.

### 3. **Precision Medicine Advancement**
The substantial performance gains with genomic data demonstrate the potential for personalized survival prediction, moving beyond traditional clinical staging toward molecular-based prognostication.

---

## Technical Implementation

### Dependencies
```
- Python 3.8+
- scikit-survival
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- lifelines (for comparison)
```

### Data Processing Pipeline
1. **Data Loading & Preprocessing**
   - Handle missing values and censoring
   - Feature encoding and standardization
   - Survival data structure creation

2. **Feature Engineering**
   - PCA dimensionality reduction
   - Clinical variable processing
   - Genomic feature selection

3. **Model Training & Evaluation**
   - Cross-validation framework
   - Survival metric calculation (C-index, IBS)
   - Statistical significance testing

### Evaluation Metrics
- **Concordance Index (C-Index):** Measures discrimination ability
- **Integrated Brier Score (IBS):** Measures calibration accuracy
- **Hazard Ratios:** Risk quantification for Cox regression

---

## Repository Structure
```
├── data/
│   ├── metabric_clinical_data.csv
│   ├── metabric_expression_data.csv
│   └── metabric_mutation_data.csv
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_reduction_pca.ipynb
│   ├── 03_traditional_survival_analysis.ipynb
│   └── 04_machine_learning_survival_models.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── survival_models.py
│   ├── evaluation_metrics.py
│   └── visualization.py
├── results/
│   ├── model_performance_comparison.csv
│   ├── hazard_ratios_table.csv
│   └── survival_curves.png
└── presentation/
    └── METABRIC_Final_Presentation.pdf
```

---

## Keywords
survival analysis, breast cancer, machine learning, genomics, Random Survival Forest, Cox regression, METABRIC, precision medicine, high-dimensional data

---

## Citation
If you use this work, please cite:
```
Everaert, M. & Holt, D. (2025). Survival Analysis and Predictive Modeling in Breast Cancer: 
Integrating Clinical and Genomic Features Using Machine Learning. Graduate Mathematics Final Project.
```

## Acknowledgments
- METABRIC Consortium for dataset access
- Astley et al. (2024) for survival analysis framework reference
- Course instructors and peers for guidance and feedback