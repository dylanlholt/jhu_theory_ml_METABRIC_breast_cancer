# %%
import pandas as pd 
import numpy as np 

# %%
# Import breast cancer dataset
data = pd.read_csv('../data/METABRIC_RNA_Mutation.csv')

# %%
data.shape

# %%
# Clinical Indicators EDA
data['age_at_diagnosis'].plot(kind='hist',title='Age at Diagnosis')

# %%
data['tumor_size'].plot(kind='hist',title='Tumor Size')

# %%
data['lymph_nodes_examined_positive'].plot(kind='hist',title='Lymph Node Involvment')

# %%
data['mutation_count'].plot(kind='hist',title='Number of Relevant Gene Mutations')

# %%
data[['age_at_diagnosis','tumor_size','lymph_nodes_examined_positive','mutation_count']].describe()

# %%
data['chemotherapy'].value_counts().plot(kind='barh',title='Chemotherapy 1=Yes/0=No')

# %%
data['hormone_therapy'].value_counts().plot(kind='barh',title='Hormonal Treatment 1=Yes/0=No')

# %%
data['type_of_breast_surgery'].value_counts().plot(kind='barh',title='Type of Breast Surgery')

# %%
data['type_of_breast_surgery'].describe()

# %%
mrna_score_columns = [x for x in data.columns[31:] if '_mut' not in x]
len(mrna_score_columns)

# %%
mrna_medians = data[mrna_score_columns].describe().loc['50%']
mrna_medians.plot(kind='hist',title='Median mRNA Z-Scores')

# %%
mrna_means = data[mrna_score_columns].describe().loc['mean']
mrna_means.plot(kind='hist',title='Mean mRNA Z-Scores')

# %%
mrna_medians.describe()

# %%
mrna_means.describe()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
clin_corr = data[['overall_survival_months','age_at_diagnosis','tumor_size','lymph_nodes_examined_positive','mutation_count']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(clin_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Clinical Indicators Correlation with Survival Time')
plt.show()

# %%
# Survival Snapshot
# data['death_from_cancer'].value_counts()
death_stage = pd.crosstab(data['death_from_cancer'],data['tumor_stage'])
# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(death_stage, annot=True, cmap=['white'], fmt=".2f", cbar=False)
plt.title('Disease-Specific Mortality by Tumor Stage')
plt.show()

# %%
# Survival Snapshot
# data['death_from_cancer'].value_counts()
pd.crosstab(data['death_from_cancer'],data['tumor_stage'])

# %%
data['overall_survival_months'].plot(kind='hist')

# %%
# Survival Snapshot
# data['death_from_cancer'].value_counts()
pd.crosstab(data['death_from_cancer'],data['tumor_stage'])


