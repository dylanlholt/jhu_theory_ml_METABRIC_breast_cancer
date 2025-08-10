# %%
#Group project final EDA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# %%
data = pd.read_csv('METABRIC_RNA_Mutation.csv')
cols = list(data.columns)

# %%
data.head()

# %% [markdown]
# ### Survival Months by Tumor Size and Type of Surgery ###

# %%
np.unique(list(data['type_of_breast_surgery']))

# %%
fig, ax = plt.subplots()
ax.scatter(data[data['type_of_breast_surgery']=='BREAST CONSERVING']['tumor_size'], data[data['type_of_breast_surgery']=='BREAST CONSERVING']['overall_survival_months'], color='k', label='Conserving')
ax.scatter(data[data['type_of_breast_surgery']=='MASTECTOMY']['tumor_size'], data[data['type_of_breast_surgery']=='MASTECTOMY']['overall_survival_months'], label='Mastectomy', color='salmon', alpha=0.4)
ax.legend(loc='upper right')
ax.set_title('Survival Months by Tumor Size and Type of Surgery')
ax.set_ylabel('overall_survival_months')
ax.set_xlabel('tumor_size')


# %% [markdown]
# ### Density Estimate of Survival Months by Tumor Stage ###

# %%
sns.displot(data, x="overall_survival_months", hue="tumor_stage", kind="kde", cut=0, palette='dark')
plt.title('KDE Distributions of overall_survival_months by tumor_stage')
plt.xlabel('overall_survival_months')
plt.ylabel('Density')

# %% [markdown]
# ### Gene Mutation Data on Survival Months and Death from Cancer ###
# In this section we look at several simple models to predict the overall_survival_months as well as death_from_cancer features. First the models are implemented without the gene mutation data and only on clinical variables such as age_at_diagnosis and tumor_size. Next we employ PCA on the gene mutation data and include the PCs in the regression analysis to see how much information the gene mutations contain for the two dependent variables. We also look at models with just a handful of gene mutations that appear to be the most relevant.

# %%
#Multiple Linear Regression without gene mutation data
data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)

Y1 = data['overall_survival_months']
Y2 = data['death_from_cancer']
X1 = data[['age_at_diagnosis', 'mutation_count', 'neoplasm_histologic_grade', 'nottingham_prognostic_index', 'tumor_size', 'tumor_stage', 'radio_therapy', 'chemotherapy', 'hormone_therapy', 'lymph_nodes_examined_positive']]

#Multiple linear regression
X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.2, random_state=42)

#Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

#Predict on the test set
y_pred = model.predict(X_test)

#Model output
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

#Using statsmodels to get summary table
X_sm = sm.add_constant(X1)
model_sm = sm.OLS(Y1, X_sm).fit()
#Print summary table
print(model_sm.summary())



# %% [markdown]
# There are a number of variables included in the regression that are not statistically significant such as neoplasm_histologic_grade. The MSE of this model on the test set was 5,472 and it has an R2 of 0.15.

# %%
np.unique(y_train2)

# %%
#Logistic regression for the death_from_cancer variable using the same features as above
X_train2, X_test2, y_train2, y_test2 = train_test_split(X1, Y2, test_size=0.3, random_state=42)

# 4. Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train2, y_train2)

# 5. Predict on the test set
y_pred2 = model.predict(X_test2)

# 6. Evaluate the model
print("Accuracy:", accuracy_score(y_test2, y_pred2))
print("\nClassification Report:\n", classification_report(y_test2, y_pred2))
print("\nConfusion Matrix:\n", confusion_matrix(y_test2, y_pred2))

# Optional: Check model coefficients
print("\nIntercept:", model.intercept_)
print("Coefficients:", model.coef_)

# %%
cm = confusion_matrix(y_test2, y_pred2)

# Define class names (0 = No, 1 = Yes)
labels = ['Death from Cancer', 'Death from Other', 'Living']

# Create heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# %% [markdown]
# Now we include the gene data after applying PCA to see if the gene mutation data will contribute to the accuracy of predictions.
# 

# %%
columns = ['pik3ca','tp53','muc16','ahnak2','kmt2c','syne1','gata3','map3k1','ahnak','dnah11','cdh1','dnah2','kmt2d','ush2a','ryr2']   
fig, axs = plt.subplots(3, 5, figsize=(15, 10))
fig.suptitle('Survival of patients with some of gene mutations.')

for i,ax in zip(data.loc[:,columns].columns,axs.flatten()):
    sns.histplot(data[i][data['death_from_cancer']!='Died of Disease'], color='g', label = 'Survived',ax=ax, stat='percent', binwidth=0.25)
    sns.histplot(data[i][data['death_from_cancer']=='Died of Disease'], color='r', label = 'Died',ax=ax, stat='percent', binwidth=0.25)
    ax.legend(loc='best')
plt.tight_layout()
plt.show()

# %%
gene_data = data.iloc[:, 31:-173]
gene_data.head()

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def perform_pca(df, n_components):
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)

    # Create a DataFrame with the principal components
    pc_columns = [f'PC{i+1}' for i in range(n_components)]
    pc_df = pd.DataFrame(data=principal_components, columns=pc_columns)

    return pc_df, pca.explained_variance_ratio_


pc_df, var_expl = perform_pca(gene_data, 100)

# %%
def plot_pca_variance(df):
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Apply PCA with all components
    pca = PCA()
    pca.fit(scaled_data)

    # Explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = explained_variance_ratio.cumsum()

    # Scree plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_pca_variance(gene_data)

# %%
pc_df.to_csv('PC_full.csv')

# %%
data.reset_index(inplace=True, drop=True)
pc_df.reset_index(inplace=True, drop=True)

# %%
data_pcs = pd.concat([data, pc_df], axis=1)

# %%
#Logistic Regression with first 25 PCs
#Logistic regression for the death_from_cancer variable using the same features as above
X3 = data_pcs[['age_at_diagnosis', 'mutation_count', 'neoplasm_histologic_grade', 'nottingham_prognostic_index', 'tumor_size', 'tumor_stage', 'radio_therapy', 'chemotherapy', 'hormone_therapy', 'lymph_nodes_examined_positive', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18', 'PC19', 'PC20', 'PC21', 'PC22', 'PC23', 'PC24', 'PC25']]


X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, Y2, test_size=0.3, random_state=42)

# 4. Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train3, y_train3)

# 5. Predict on the test set
y_pred3 = model.predict(X_test3)

# 6. Evaluate the model
print("Accuracy:", accuracy_score(y_test3, y_pred3))
print("\nClassification Report:\n", classification_report(y_test3, y_pred3))
print("\nConfusion Matrix:\n", confusion_matrix(y_test3, y_pred3))

# Optional: Check model coefficients
print("\nIntercept:", model.intercept_)
print("Coefficients:", model.coef_)


# %%
cm = confusion_matrix(y_test3, y_pred3)

# Define class names
labels = ['Death from Cancer', 'Death from Other', 'Living']

# Create heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# %%
feature_names = X3.columns
coefficients = model.coef_[0]

#Contribution of each feature (coefficient)
contrib_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', key=abs, ascending=False)

print(contrib_df)

# %%
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, train_test_split

# Split into training and test sets (optional but recommended)
X_train, X_test, y_train, y_test = train_test_split(X3, y, test_size=0.2, random_state=42)

# Base model
model = LinearRegression()

# Store performance for different feature counts
feature_counts = []
scores = []

# Try different numbers of features (from all to just 1)
for k in range(1, X_train.shape[1] + 1):
    rfe = RFE(model, n_features_to_select=k)
    X_train_rfe = rfe.fit_transform(X_train, y_train)
    
    # Evaluate with cross-validation
    score = cross_val_score(model, X_train_rfe, y_train, cv=5, scoring='r2').mean()
    
    feature_counts.append(k)
    scores.append(score)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(feature_counts, scores, marker='o')
plt.xlabel("Number of Features")
plt.ylabel("Cross-Validated R² Score")
plt.title("R² Score vs. Number of Features")
plt.grid(True)
plt.show()

# Find best number of features
best_k = feature_counts[np.argmax(scores)]
print(f"Best number of features: {best_k}")
print(f"Best cross-validated R²: {max(scores):.4f}")

# Fit final model with best features
final_rfe = RFE(model, n_features_to_select=best_k)
final_rfe.fit(X_train, y_train)

selected_features = X_train.columns[final_rfe.support_]
print("Selected features:", list(selected_features))

# Train and evaluate on test set
X_test_selected = final_rfe.transform(X_test)
model.fit(final_rfe.transform(X_train), y_train)
test_score = model.score(X_test_selected, y_test)
print(f"Test R² with selected features: {test_score:.4f}")

# %%
#Linear Regression with the PCs
#Y1 = data['overall_survival_months']
#X3

#Multiple linear regression
X_train, X_test, y_train, y_test = train_test_split(X3, Y1, test_size=0.2, random_state=42)

#Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

#Predict on the test set
y_pred = model.predict(X_test)

#Model output
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

#Using statsmodels to get summary table
X_sm = sm.add_constant(X3)
model_sm = sm.OLS(Y1, X_sm).fit()
#Print summary table
print(model_sm.summary())

# %%
X3.iloc[:, 10:]

# %%
#Regression of just gene mutation PCs
X4 = X3.iloc[:, 10:]

X4_sm = sm.add_constant(X4)
model_sm = sm.OLS(Y1, X4_sm).fit()
#Print summary table
print(model_sm.summary())


# %%



