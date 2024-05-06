# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports for model building and evaluation
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Specific model imports from scikit-learn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# XGBoost
import xgboost as xgb
from xgboost import XGBClassifier

# Plotly for interactive plotting
import plotly.graph_objects as go

from scipy.stats import f_oneway, chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import mannwhitneyu

hd_df = pd.read_csv("../Data/heart_disease_uci.csv")
hd_df.drop_duplicates(inplace=True)

print(hd_df.shape)
print(hd_df.head(2))
print(hd_df.info())
print("Total NaN:\n", hd_df.isnull().sum())
total = hd_df.isnull().sum().sort_values(ascending=False)
total_select = total.head(20)

# Create the plot with Plotly
# fig = px.bar(total_select,
#              x=total_select.index,
#              y=total_select.values,
#              labels={'x': 'Columns', 'y': 'Count'},
#              title='Total Missing Values')

# Update layout
# fig.update_layout(
#     xaxis_title='Columns',
#     yaxis_title='Count',
#     title={'text': 'Total Missing Values', 'x': 0.5, 'xanchor': 'center'},
#     font=dict(size=20)
# )

# Show plot
# fig.show()

# Rename columns, easier to remember
hd_df = hd_df.rename(columns={"cp": "chest_pain_type", "trestbps": "resting_blood_pressure",
                              "chol": "serum_cholesterol", "fbs": "bloodsugar_above_120",
                              "restecg": "resting_ecg_results",
                              "thalch": "maximum_heart_rate", "exang": "st_depress_ex_rel_rest", "slope": "st_slope",
                              "ca": "num_vessels_fluoro",
                              "thal": "thallium_test", "num": "pre_class"})

# Category data exploratory
print("sex : ", hd_df.sex.unique().tolist())
print("dataset : ", hd_df.dataset.unique().tolist())
print("chest_pain_type : ", hd_df.chest_pain_type.unique().tolist())
print("bloodsugar_above_120 : ", hd_df.bloodsugar_above_120.unique().tolist())  # 59 missing
print("resting_ecg_results : ", hd_df.resting_ecg_results.unique().tolist())  # 2 are missing
print("st_depress_ex_rel_rest : ", hd_df.st_depress_ex_rel_rest.unique().tolist())  # 55 are missing
print("st_slope : ", hd_df.st_slope.unique().tolist())  # 309 are missing
print("thallium_test : ", hd_df.thallium_test.unique().tolist())  # 486 are missing
print("num_vessels_fluoro : ", hd_df.num_vessels_fluoro.unique().tolist())  # 611 are missing
print("pre_class : ", hd_df.pre_class.unique().tolist())

# Numeric Data info
hd_df[["age", "resting_blood_pressure", "serum_cholesterol", "maximum_heart_rate", "oldpeak"]].describe()

hd_num = hd_df.select_dtypes(include=['float64', 'int64'])
hd_num_corr = hd_num.corr()['pre_class'][:-1]  # -1 removes the last as it would be 1...

# Plot hd_num_corr
# plt.figure(figsize=(10, 6))
# hd_num_corr.plot(kind='bar', color='skyblue')
# plt.title('Correlation with pre_class')
# plt.xlabel('Numerical Features')
# plt.ylabel('Correlation Coefficient')
# plt.xticks(rotation=45, ha='right')
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

top_features = hd_num_corr[abs(hd_num_corr) > 0.5].sort_values(ascending=False)  # displays pearsons correlation coefficient greater than 0.5
print("There is {} strongly correlated values with pre_class:\n{}".format(len(top_features), top_features))
exit()

# # Plot histogram
# plt.figure(figsize=(8, 6))
# plt.hist(hd_df['pre_class'], bins=5, density=True, color='skyblue', alpha=0.7)
# plt.title('Histogram of pre_class')
# plt.xlabel('pre_class')
# plt.ylabel('Probability')
# plt.grid(linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()
# converting "nan" str to numpy NaN.
hd_df["num_vessels_fluoro"] = np.where(hd_df["num_vessels_fluoro"] == 'nan', np.nan,
                                       hd_df["num_vessels_fluoro"].astype(float))

len_df = len(hd_df)
for column in hd_df.columns:
    print("% of missing values for column ", column, " ", round(hd_df[column].isnull().sum() * 100 / len_df, 3))

# Numeric missing value columns: Use median imputation
numeric_columns = ["resting_blood_pressure", "serum_cholesterol", "maximum_heart_rate", "oldpeak"]
for column in numeric_columns:
    mean_value = np.nanmean(hd_df[column])
    print("Changing NaN with mean:", column, round(mean_value, 2))
    hd_df[column] = np.where(np.isnan(hd_df[column]), mean_value, hd_df[column])

columns_of_interest = ["sex", "bloodsugar_above_120", "resting_ecg_results", "st_depress_ex_rel_rest", "st_slope",
                       "thallium_test", "num_vessels_fluoro"]

missing_percentages = []

for column in columns_of_interest:
    missing_percentage = hd_df[column].isnull().sum() * 100 / len(hd_df)
    print(column, "missing value percentage:", missing_percentage)
    missing_percentages.append(missing_percentage)

# Heatmap to show correlations of missingness between columns -> thallium and num_vessel_fluoro are mostly missing together
# category_missing_df = hd_df[["bloodsugar_above_120", "resting_ecg_results", "st_depress_ex_rel_rest", "st_slope", "thallium_test", "num_vessels_fluoro"]]
# msno.heatmap(category_missing_df)
# plt.show()

# filling the missing with the most frequent value per category
# Categorical columns with mode
categorical_missing_columns = ["bloodsugar_above_120", "resting_ecg_results", "st_depress_ex_rel_rest"]
categorical_imputer = SimpleImputer(strategy='most_frequent')
hd_df[categorical_missing_columns] = categorical_imputer.fit_transform(hd_df[categorical_missing_columns])
# Fill missing values with 'Unknown', I like to check if this feature has strong relationship between target.
hd_df['st_slope'] = hd_df['st_slope'].fillna('Unknown')
hd_df['thallium_test'] = hd_df['thallium_test'].fillna('Unknown')
hd_df['num_vessels_fluoro'] = hd_df['num_vessels_fluoro'].fillna('Unknown')
hd_df["num_vessels_fluoro"] = hd_df.num_vessels_fluoro.astype(str)

print("\nChecking null values again:\n")
print(hd_df.isnull().sum())

analysis_hd_df = hd_df.copy()

# if 'age' in analysis_hd_df.columns and 'pre_class' in analysis_hd_df.columns:
#     # 5 years interval
#     bins = list(range(analysis_hd_df['age'].min(), analysis_hd_df['age'].max() + 6, 5))
#     labels = [f"{i}-{i + 4}" for i in range(analysis_hd_df['age'].min(), analysis_hd_df['age'].max(), 5)]
#     analysis_hd_df['age_group'] = pd.cut(analysis_hd_df['age'], bins=bins, labels=labels, right=False)
#
#     # Bianry classification : Heart Disease/ No Heart Disease
#     analysis_hd_df['has_disease'] = analysis_hd_df['pre_class'].apply(
#         lambda x: 'Heart Disease' if x > 0 else 'No Heart Disease')
#
#     # Number Count
#     age_group_counts = analysis_hd_df.groupby(['age_group', 'has_disease']).size().unstack(fill_value=0)
#     age_group_counts['Total'] = age_group_counts.sum(axis=1)
#     age_group_counts['Heart Disease Ratio'] = (
#             (age_group_counts['Heart Disease'] / age_group_counts['Total']) * 100).round(2)
#
#     # bar Plot
#     fig = go.Figure()
#     fig.add_trace(go.Bar(
#         x=age_group_counts.index,
#         y=age_group_counts['No Heart Disease'],
#         name='No Heart Disease'
#     ))
#     fig.add_trace(go.Bar(
#         x=age_group_counts.index,
#         y=age_group_counts['Heart Disease'],
#         name='Heart Disease',
#         text=age_group_counts['Heart Disease Ratio'].apply(lambda x: f"{x}%"),
#         textposition='inside'
#     ))
#
#     fig.update_layout(
#         barmode='stack',
#         title="Age Distribution and Ratio of Heart Disease in Each Group",
#         xaxis_title='Age',
#         yaxis_title='Count',
#         legend_title='Status',
#         xaxis={'categoryorder': 'array', 'categoryarray': labels}
#     )
#
#     fig.show()
# else:
#     print("Ensure 'age' and  'pre_class' in df")

# analysis_hd_df['Heart Disease'] = analysis_hd_df['pre_class'].apply(lambda x: 'No Heart Disease' if x == 0 else 'Heart Disease')
# plot_data = analysis_hd_df.groupby(['sex', 'Heart Disease']).size().reset_index(name='count')
# total_counts = plot_data.groupby('sex')['count'].sum().reset_index(name='total')
# plot_data = plot_data.merge(total_counts, on='sex')
# plot_data['ratio'] = plot_data.apply(lambda x: x['count'] / x['total'] if x['Heart Disease'] == 'Heart Disease' else 0, axis=1)
# plot_data['text'] = plot_data.apply(lambda x: f"{x['ratio']:.2%}" if x['Heart Disease'] == 'Heart Disease' and x['ratio'] != 0 else '', axis=1)
#
# # Sort plot_data to ensure 'Heart Disease' appears on top
# plot_data = plot_data.sort_values(by=['Heart Disease', 'sex'], ascending=[True, True])
#
# # Plotting using Plotly
# fig = px.bar(plot_data, x='sex', y='count', color='Heart Disease',
#              color_discrete_map={'No Heart Disease': 'skyblue', 'Heart Disease': 'salmon'},
#              text='text', category_orders={"Heart Disease": ["No Heart Disease", "Heart Disease"]})
#
# # Update the layout and annotations
# fig.update_layout(
#     title='Heart Disease Status by Sex',
#     xaxis_title='Sex',
#     yaxis_title='Count',
#     barmode='stack'
# )
# fig.update_traces(texttemplate='%{text}', textposition='outside')
# fig.show()
#
# # Ensure 'pre_class' is treated as a category for proper grouping
# analysis_hd_df['pre_class'] = pd.Categorical(hd_df['pre_class'], categories=[0, 1, 2, 3, 4], ordered=True)
#
# # Grouping data and counting occurrences
# plot_data = analysis_hd_df.groupby(['dataset', 'pre_class']).size().reset_index(name='count')
#
# # Plotting using Plotly
# fig = px.bar(plot_data, x='dataset', y='count', color='pre_class', barmode='group',
#              category_orders={"pre_class": [0, 1, 2, 3, 4]},
#              labels={'pre_class': 'Pre-Class', 'dataset': 'Dataset'})
#
# # Update layout and display the figure
# fig.update_layout(
#     title='Distribution of Different Type of Heart Disease Across Datasets',
#     xaxis_title='Dataset',
#     yaxis_title='Count',
#     legend_title='Heart Disease Types'
# )
# fig.show()
#
# # Ensure 'pre_class' is treated as a category for proper grouping
# analysis_hd_df['pre_class'] = pd.Categorical(hd_df['pre_class'], categories=[0, 1, 2, 3, 4], ordered=True)
#
# # Grouping data and counting occurrences
# plot_data = analysis_hd_df.groupby(['dataset', 'pre_class']).size().reset_index(name='count')
#
# # Plotting using Plotly
# fig = px.bar(plot_data, x='dataset', y='count', color='pre_class', barmode='group',
#              category_orders={"pre_class": [0, 1, 2, 3, 4]},
#              labels={'pre_class': 'Pre-Class', 'dataset': 'Dataset'})
#
# # Update layout and display the figure
# fig.update_layout(
#     title='Distribution of Different Type of Heart Disease Across Datasets',
#     xaxis_title='Dataset',
#     yaxis_title='Count',
#     legend_title='Heart Disease Types'
# )
# fig.show()
#
# # Ensure 'pre_class' is treated as a category for proper grouping
# analysis_hd_df['pre_class'] = pd.Categorical(hd_df['pre_class'], categories=[0, 1, 2, 3, 4], ordered=True)
# # Grouping data and counting occurrences
# plot_data = analysis_hd_df.groupby(['chest_pain_type', 'pre_class']).size().reset_index(name='count')
# # Plotting using Plotly
# fig = px.bar(plot_data, x='chest_pain_type', y='count', color='pre_class', barmode='group',
#              category_orders={"pre_class": [0, 1, 2, 3, 4]},
#              labels={'pre_class': 'Different Type of Heart Disease', 'chest_pain_type': 'Type of Chest Pain'})
# # Update layout and display the figure
# fig.update_layout(
#     title='Distribution of Heart Disease Across Different Type of Chest Pain',
#     xaxis_title='Type of Chest Pain',
#     yaxis_title='Count',
#     legend_title='Heart Disease'
# )
# fig.show()
#
#
# # calculating chi squared
# analysis_hd_df['Chest Pain'] = analysis_hd_df['chest_pain_type'].replace({
#     'typical angina': 'chest pain',
#     'non-anginal': 'chest pain',
#     'atypical angina': 'chest pain',
#     'asymptomatic': 'no chest pain'
# })
#
# # Contingency Table for Chi-Square Test
# contingency_table = pd.crosstab(analysis_hd_df['Chest Pain'], analysis_hd_df['Heart Disease'])
# chi2, p_value, dof, expected = chi2_contingency(contingency_table)
#
# # Output the results
# print("Chi-squared Test statistic:", chi2)
# print("P-value:", p_value)
#
# # Using Plotly to plot the data
# fig = px.bar(contingency_table,
#              title="Chest Pain Type vs. Heart Disease Status",
#              labels={'value': 'Number of Patients', 'Chest Pain': 'Chest Pain Type'},
#              orientation='v',
#              barmode='stack')
#
# fig.update_xaxes(title_text='Chest Pain Type')
# fig.update_yaxes(title_text='Number of Patients')
# fig.update_layout(bargap=0.2)  # Optional: adjust the gap between bars
# fig.show()

avg_blood_pressure = analysis_hd_df.groupby('pre_class')['resting_blood_pressure'].mean().reset_index()
#
# # Creating the bar plot
# fig = px.bar(avg_blood_pressure, x='pre_class', y='resting_blood_pressure',
#              text='resting_blood_pressure', title='Average Resting Blood Pressure by Different Type of Heart Disease')
# fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
# fig.show()

# ANOVA test across all groups
anova_result = f_oneway(
    *[group['resting_blood_pressure'].values for name, group in analysis_hd_df.groupby('pre_class')])
print(f'\n\nANOVA result: F-statistic = {anova_result.statistic:.2f}, p-value = {anova_result.pvalue:.4f}')

# Conducting post-hoc analysis if ANOVA is significant
if anova_result.pvalue < 0.05:
    print("Significant differences were found, performing post-hoc testing...")
    # Performing Tukey's HSD test
    tukey_result = pairwise_tukeyhsd(endog=analysis_hd_df['resting_blood_pressure'],
                                     groups=analysis_hd_df['pre_class'],
                                     alpha=0.05)
    print(tukey_result)
else:
    print("No significant differences were found among groups.")

analysis_hd_df['Heart Disease'] = analysis_hd_df['pre_class'].map(
    lambda k: 'No Heart Disease' if k == 0 else 'Heart Disease')

# Calculating the average resting blood pressure for each category
avg_blood_pressure = analysis_hd_df.groupby('Heart Disease')['resting_blood_pressure'].mean().reset_index()
#
# # Creating the bar plot
# fig = px.bar(avg_blood_pressure, x='Heart Disease', y='resting_blood_pressure',
#              text='resting_blood_pressure', title='Average Resting Blood Pressure by Heart Disease Status')
# fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
# fig.update_xaxes(title_text='Resting Blood Pressure')
# fig.update_yaxes(title_text='Number of Patients')
# fig.show()

data = analysis_hd_df['resting_blood_pressure']

# Creating the histogram
# fig = px.histogram(data, nbins=30, title='Histogram of Resting Blood Pressure')
# fig.update_layout(bargap=0.1)
# fig.show()

# Separating the groups
group_no_disease = analysis_hd_df[analysis_hd_df['Heart_Disease'] == 'No Heart Disease']['resting_blood_pressure']
group_with_disease = analysis_hd_df[analysis_hd_df['Heart_Disease'] == 'Heart Disease']['resting_blood_pressure']
#
# # Performing the Mann-Whitney U test because distribution is skewed to the right
u_stat, p_val = mannwhitneyu(group_no_disease, group_with_disease)
#
# print(f'Mann-Whitney U test results: U-statistic = {u_stat:.3f}, p-value = {p_val:.4f}')
# avg_serum_cholesterol = analysis_hd_df.groupby('pre_class')['serum_cholesterol'].mean().reset_index()

# # Creating the bar plot
# fig = px.bar(avg_serum_cholesterol, x='pre_class', y='serum_cholesterol',
#              text='serum_cholesterol', title='Average Serum Cholesterol by Different type of Heart Disease')
# fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
# fig.update_xaxes(title_text='Different type of Heart Disease')
# fig.update_yaxes(title_text='Serum Cholesterol')
# fig.show()

# ANOVA test across all groups
anova_result = f_oneway(*[group['serum_cholesterol'].values for name, group in analysis_hd_df.groupby('pre_class')])
print(f'\n\nCholesterol ANOVA result: F-statistic = {anova_result.statistic:.2f}, p-value = {anova_result.pvalue:.4f}')

# Conducting post-hoc analysis if ANOVA is significant
if anova_result.pvalue < 0.05:
    print("Significant differences were found, performing post-hoc testing...")
    # Performing Tukey's HSD test
    tukey_result = pairwise_tukeyhsd(endog=analysis_hd_df['resting_blood_pressure'],
                                     groups=analysis_hd_df['pre_class'],
                                     alpha=0.05)
    print(tukey_result)
else:
    print("No significant differences were found among groups.")

# Plotting the graph with solid colors
# fig = px.bar(analysis_hd_df, x='pre_class', color='bloodsugar_above_120',
#              title="Blood Sugar Levels by Different Type of Heart Disease",
#              labels={'pre_class': 'Type of Heart Disease'},
#              barmode='group',
#              color_discrete_map={True: 'skyblue', False: 'salmon'})  # Define specific colors
# fig.update_layout(xaxis_title='Different Type of Heart Disease', yaxis_title='Count',
#                   legend_title='Blood Sugar > 120')  # Setting the background to white
# fig.update_traces(marker_line_width=0)  # Remove line around bars
# fig.show()

# Ensure 'pre_class' is treated as a category for proper grouping
analysis_hd_df['pre_class'] = pd.Categorical(hd_df['pre_class'], categories=[0, 1, 2, 3, 4], ordered=True)

# Grouping data and counting occurrences
plot_data = analysis_hd_df.groupby(['resting_ecg_results', 'pre_class'], observed=False).size().reset_index(
    name='count')

# Plotting using Plotly
# fig = px.bar(plot_data, x='resting_ecg_results', y='count', color='pre_class', barmode='group',
#              category_orders={"pre_class": [0, 1, 2, 3, 4]},
#              labels={'pre_class': 'Pre-Class', 'resting_ecg_results': 'Resting ECG Result'})
#
# Update layout and display the figure
# fig.update_layout(
#     title='Distribution of Heart Disease Across Resting ECG Result',
#     xaxis_title='Resting ECG Result',
#     yaxis_title='Count',
#     legend_title='Heart Disease'
# )
# fig.show()

# see if exercise-induced angina is true/false in the different type of heart disease
# fig = px.bar(analysis_hd_df, x='pre_class', color='st_depress_ex_rel_rest',
#              title="Exercise-Induced Angina by Different Type of Heart Disease",
#              labels={'pre_class': 'Pre_class'},
#              barmode='group',
#              color_discrete_map={True: 'skyblue', False: 'salmon'})  # Define specific colors
# fig.update_layout(xaxis_title='Different Type of Heart Disease', yaxis_title='Count',
#                   legend_title='Exercise-Induced Angina')  # Setting the background to white
# fig.update_traces(marker_line_width=0)  # Remove line around bars
# fig.show()

# Feature engineering
# 'sex', 'bloodsugar_above_120','st_depress_ex_rel_rest' are binary columns, I decide convert them to 1 and 0
hd_df['sex'] = hd_df['sex'].map({'Male': 1, 'Female': 0})
# Convert to boolean first if they are strings that read 'True'/'False'
hd_df['bloodsugar_above_120'] = hd_df['bloodsugar_above_120'].astype(bool)
hd_df['st_depress_ex_rel_rest'] = hd_df['st_depress_ex_rel_rest'].astype(bool)

# Then apply your original mapping
hd_df['bloodsugar_above_120'] = hd_df['bloodsugar_above_120'].map({True: 1, False: 0})
hd_df['st_depress_ex_rel_rest'] = hd_df['st_depress_ex_rel_rest'].map({True: 1, False: 0})

# Encode categorical columns
hd_df_dummies = pd.get_dummies(hd_df, columns=["dataset", "chest_pain_type", "resting_ecg_results",
                                               "st_slope", "thallium_test", "num_vessels_fluoro"], drop_first=False)

# Numeric Columns
columns = ["resting_blood_pressure", "serum_cholesterol", "maximum_heart_rate", "oldpeak", "age"]

# Create a figure
fig = go.Figure()

# Add a box plot for each column
for col in columns:
    fig.add_trace(go.Box(y=hd_df_dummies[col], name=col))

# Update layout for a better fit
fig.update_layout(
    title="Distribution of Various Health Metrics",
    yaxis_title="Values",
    xaxis_title="Health Metrics",
    showlegend=False,
    boxmode='group'  # group together boxes of the different traces for each column
)


# Show the figure
# fig.show()


# Function to calculate outliers
def count_outliers(data_, feature_):
    Q1_ = data_[feature_].quantile(0.25)
    Q3_ = data_[feature_].quantile(0.75)
    IQR_ = Q3_ - Q1_
    lower_bound = Q1_ - 1.5 * IQR_
    upper_bound = Q3_ + 1.5 * IQR_
    outliers = data_[(data_[feature_] < lower_bound) | (data_[feature_] > upper_bound)]
    return outliers.shape[0]


# Columns to check for outliers
columns = ["resting_blood_pressure", "serum_cholesterol", "maximum_heart_rate", "oldpeak", "age"]

# DataFrame 'hd_df' assumed to be previously defined and loaded
outlier_counts = {column: count_outliers(hd_df_dummies, column) for column in columns}
print(outlier_counts)

q_low = hd_df_dummies['serum_cholesterol'].quantile(0.05)
q_high = hd_df_dummies['serum_cholesterol'].quantile(0.95)

hd_df_dummies['serum_cholesterol'] = np.where(hd_df_dummies['serum_cholesterol'] < q_low, q_low,
                                              hd_df_dummies['serum_cholesterol'])
hd_df_dummies['serum_cholesterol'] = np.where(hd_df_dummies['serum_cholesterol'] > q_high, q_high,
                                              hd_df_dummies['serum_cholesterol'])

# Apply IQR
numerical_columns = ["resting_blood_pressure", "maximum_heart_rate", "oldpeak", "serum_cholesterol", "age"]

for col in numerical_columns:
    Q1 = hd_df_dummies[col].quantile(0.25)
    Q3 = hd_df_dummies[col].quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR
    hd_df_dummies = hd_df_dummies[(hd_df_dummies[col] <= upper_limit) & (hd_df_dummies[col] >= lower_limit)]

# Features to scale
features_to_scale = ["resting_blood_pressure", "maximum_heart_rate", "oldpeak", "serum_cholesterol", "age"]

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the selected features
hd_df_dummies[features_to_scale] = scaler.fit_transform(hd_df_dummies[features_to_scale])

# Create a list of float column names to check for skewing
float_cols = ["resting_blood_pressure", "maximum_heart_rate", "oldpeak", "serum_cholesterol", "age"]

# Define a skew limit above which we will consider transforming
skew_limit = 0.75

# Calculate skewness values for these columns
skew_vals = hd_df_dummies[float_cols].skew()

print(skew_vals)

# Showing the skewed columns
skew_cols = (skew_vals
             .sort_values(ascending=False)
             .to_frame()
             .rename(columns={0: 'Skew'})
             .query('abs(Skew) > {}'.format(skew_limit)))

print(skew_cols)

# Creating two subplots and a figure using matplotlib
fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))

# Creating a histogram on the ax_before subplot
hd_df_dummies["oldpeak"].hist(ax=ax_before)

# Apply a log transformation (using numpy) to this column
hd_df_dummies["oldpeak"].apply(np.log1p).hist(ax=ax_after)

# Formatting titles, labels for each subplot
ax_before.set(title='Before np.log1p', ylabel='Frequency', xlabel='Value')
ax_after.set(title='After np.log1p', ylabel='Frequency', xlabel='Value')

# Set main title for the plots
fig.suptitle('Transforming "{}" with np.log1p'.format("oldpeak"))
plt.show()

hd_df_dummies["oldpeak"] = hd_df_dummies["oldpeak"].apply(np.log1p)

# Numeric columns
features = ["resting_blood_pressure", "maximum_heart_rate", "oldpeak", "serum_cholesterol", "age"]
correlation_matrix = hd_df_dummies[features].corr()

# Plotting the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Numeric features to analyze
features = ["resting_blood_pressure", "maximum_heart_rate", "oldpeak", "serum_cholesterol", "age"]

# Perform ANOVA F-test for each numeric feature
anova_results = {}
for feature in features:
    grouped_data = [hd_df_dummies[hd_df_dummies['pre_class'] == cls][feature] for cls in
                    sorted(hd_df_dummies['pre_class'].unique())]
    f_stat, p_value = f_oneway(*grouped_data)
    anova_results[feature] = (f_stat, p_value)

# Print the results
for feature, (f_stat, p_value) in anova_results.items():
    print(f"Feature: {feature}, F-Statistic: {f_stat:.2f}, P-Value: {p_value:.4f}")

features = ['sex', 'bloodsugar_above_120', 'st_depress_ex_rel_rest', 'dataset_Cleveland', 'dataset_Hungary',
            'dataset_Switzerland', 'dataset_VA Long Beach',
            'chest_pain_type_asymptomatic', 'chest_pain_type_atypical angina', 'chest_pain_type_non-anginal',
            'chest_pain_type_typical angina',
            'resting_ecg_results_lv hypertrophy', 'resting_ecg_results_normal', 'resting_ecg_results_st-t abnormality',
            'st_slope_Unknown',
            'st_slope_downsloping', 'st_slope_flat', 'st_slope_upsloping', 'thallium_test_Unknown',
            'thallium_test_fixed defect',
            'thallium_test_normal', 'thallium_test_reversable defect', 'num_vessels_fluoro_0.0',
            'num_vessels_fluoro_1.0',
            'num_vessels_fluoro_2.0', 'num_vessels_fluoro_3.0', 'num_vessels_fluoro_Unknown']

# Initialize list to store Chi-square test results
chi_square_results = []

# Perform Chi-square test for each categorical feature
for feature in features:
    if feature in hd_df_dummies.columns:
        contingency_table = pd.crosstab(hd_df_dummies[feature], hd_df_dummies['pre_class'])
        if not contingency_table.empty:
            chi2, p, dof, ex = chi2_contingency(contingency_table)
            chi_square_results.append({'Feature': feature, 'Chi-square Statistic': chi2, 'P-value': p})
        else:
            print(f"Skipping feature '{feature}' due to empty contingency table.")
    else:
        print(f"Feature '{feature}' not found in DataFrame.")

# Create DataFrame from the results list
if chi_square_results:
    results_df = pd.DataFrame(chi_square_results)
    # Sort the DataFrame by 'Chi-square Statistic' from high to low
    sorted_results_df = results_df.sort_values(by='Chi-square Statistic', ascending=False)
    # Format the DataFrame to display four decimal places for P-value
    pd.options.display.float_format = '{:.5f}'.format
    # Print the sorted results in a column format
    print(sorted_results_df.to_string(index=False))
else:
    print("No valid chi-square results to display.")

# ML part

# Arrange hd_df_dummies dataset for training
y = hd_df_dummies["pre_class"]  # label
hd_df_dummies = hd_df_dummies.drop(["pre_class", "id"], axis=1)  # remove unecessary columns

# Sub dataset 1 - Based On chi2 test result, remove insignificant columns
hd_df_dummies_sub1 = hd_df_dummies.drop(['dataset_Switzerland', 'chest_pain_type_typical angina'], axis=1)

# Sub dataset 2 - Remove columns with > 50% missing values
hd_df_dummies_sub2 = hd_df_dummies.drop(['st_slope_Unknown', 'st_slope_downsloping', 'st_slope_flat',
                                         'st_slope_upsloping', 'thallium_test_Unknown',
                                         'thallium_test_fixed defect', 'thallium_test_normal',
                                         'thallium_test_reversable defect', 'num_vessels_fluoro_0.0',
                                         'num_vessels_fluoro_1.0', 'num_vessels_fluoro_2.0',
                                         'num_vessels_fluoro_3.0', 'num_vessels_fluoro_Unknown'], axis=1)

# Sub dataset 3 - Remove columns with > 50% missing values and insignificant columns based on chi2 test result
hd_df_dummies_sub3 = hd_df_dummies.drop(['st_slope_Unknown', 'st_slope_downsloping', 'st_slope_flat',
                                         'st_slope_upsloping', 'thallium_test_Unknown',
                                         'thallium_test_fixed defect', 'thallium_test_normal',
                                         'thallium_test_reversable defect', 'num_vessels_fluoro_0.0',
                                         'num_vessels_fluoro_1.0', 'num_vessels_fluoro_2.0',
                                         'num_vessels_fluoro_3.0', 'num_vessels_fluoro_Unknown',
                                         'dataset_Switzerland', 'chest_pain_type_typical angina'], axis=1)

# Step 1: Standardize the data
scaler = StandardScaler()
hd_df_scaled = scaler.fit_transform(hd_df_dummies)

# Step 2: Apply PCA
pca = PCA(n_components=0.95)  # keep 95% of variance
hd_df_pca = pca.fit_transform(hd_df_scaled)

# Explained variance
print(f"Explained Variance: {pca.explained_variance_ratio_}")

# Cumulative variance explained by the principal components
plt.figure(figsize=(8, 5))
plt.plot(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.title('Explained Variance by Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# Step 3: Optionally, convert back to a DataFrame
pca_columns = [f'PCA_Component_{i}' for i in range(pca.n_components_)]
hd_df_dummies_pca = pd.DataFrame(hd_df_pca, columns=pca_columns)


# Create a list of models to evaluate
def model_evaluation(df):
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=43)
    models = {
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(random_state=43),
        'Random Forest': RandomForestClassifier(random_state=43),
        'Gradient Boosting': GradientBoostingClassifier(random_state=43),
        'Support Vector Machine': SVC(random_state=43),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=43),
        'Ada Boost': AdaBoostClassifier(random_state=43),
        'XG Boost': XGBClassifier(random_state=43),

    }

    model_scores = []
    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metric = accuracy_score(y_test, y_pred)
        model_scores.append((name, metric))
        # print(name , 'mse :' , mean_squared_error(y_test , y_pred))

    sm = sorted(model_scores, key=lambda x_: x_[1], reverse=False)
    for model in sm:
        print('accuracy score of', f'{model[0]} is {model[1]}')
    print("***************************************")


for x in [hd_df_dummies, hd_df_dummies_sub1, hd_df_dummies_sub2, hd_df_dummies_sub3, hd_df_dummies_pca]:
    model_evaluation(x)

# I'll use XGBoost and RandForest

# Parameter for XGBoost
param_1 = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',  # change to 'multi:softprob' for probabilities
    'num_class': 5,
    'verbosity': 0,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
    'eval_metric': 'mlogloss'
}
param_2 = {'booster': 'gbtree',
           'objective': 'multi:softmax',
           'num_class': 5,
           'verbosity': 0,
           'colsample_bytree': 0.7,
           'eta': 0.04,
           'gamma': 0.5,
           'lambda': 5,
           'max_depth': 5,
           'min_child_weight': 5,
           'subsample': 0.6,
           'seed': 1001,
           'silent': 1,
           'nthread': 4,
           'eval_metric': 'mlogloss'}

# Parameter for Random Forest
param_rf = {
    'n_estimators': 100,
    'max_depth': 6,
    'min_samples_leaf': 3,
    'max_features': 0.7,
    'max_samples': 0.7,
    'min_impurity_decrease': 0.1,
    'bootstrap': True,
    'verbose': 0,
    'random_state': 1002,
    'n_jobs': 4
}

param_rf_2 = {
    'n_estimators': 100,
    'max_depth': 6,
    'min_samples_leaf': 4,
    'max_features': 0.7,
    'max_samples': 0.6,
    'min_impurity_decrease': 0.02,
    'bootstrap': True,
    'verbose': 0,
    'random_state': 1002,
    'n_jobs': 4
}


def xgbresult_multiclass_cv(table, label, param):
    model = None
    print('xgboost - multi-class classification with 10-fold cross-validation')

    # Reset the index of the DataFrame to avoid KeyError
    table = table.reset_index(drop=True)
    label = label.reset_index(drop=True)

    # Parameters for the XGBoost model
    params = param

    # 10-fold cross-validation setup
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=100)
    accuracy_scores = []
    f1_scores = []
    recall_scores = []
    precision_scores = []

    # Perform cross-validation
    for train_idx, test_idx in kfold.split(table, label):
        Xc_train, Xc_test = table.iloc[train_idx], table.iloc[test_idx]
        yc_train, yc_test = label[train_idx], label[test_idx]

        # Convert data into DMatrix format for XGBoost
        dtrain = xgb.DMatrix(data=Xc_train, label=yc_train)
        dtest = xgb.DMatrix(data=Xc_test, label=yc_test)

        # Train the XGBoost model with the training fold
        num_rounds = 500
        model = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_rounds)

        # Make predictions on the testing fold
        preds = model.predict(dtest)
        # For softprob, predictions need to be converted to class labels
        if params['objective'] == 'multi:softprob':
            preds = preds.argmax(axis=1)

        # Calculate the accuracy and other metrics of the current fold
        fold_accuracy = accuracy_score(yc_test, preds)
        fold_f1 = f1_score(yc_test, preds, average='weighted')
        fold_recall = recall_score(yc_test, preds, average='weighted')
        fold_precision = precision_score(yc_test, preds, average='weighted')

        accuracy_scores.append(fold_accuracy)
        f1_scores.append(fold_f1)
        recall_scores.append(fold_recall)
        precision_scores.append(fold_precision)

    # Calculate the average of each metric across all folds
    avg_accuracy = np.mean(accuracy_scores)
    avg_f1 = np.mean(f1_scores)
    avg_recall = np.mean(recall_scores)
    avg_precision = np.mean(precision_scores)

    # Print the results
    print(f"Average Accuracy: {avg_accuracy * 100:.2f}%")
    print(f"Average F1-Score: {avg_f1:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")

    return model


XGBmodel = xgbresult_multiclass_cv(hd_df_dummies, y, param_1)


def rfresult_multiclass_cv(table, label, param):
    model = None
    print('Random Forest - multi-class classification with 10-fold cross-validation')

    # Reset the index of the DataFrame to avoid KeyError
    table = table.reset_index(drop=True)
    label = label.reset_index(drop=True)

    # Parameters for the Random Forest model
    params = param

    # 10-fold cross-validation setup
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=100)
    accuracy_scores = []
    f1_scores = []
    recall_scores = []
    precision_scores = []

    # Perform cross-validation
    for train_idx, test_idx in kfold.split(table, label):
        Xc_train, Xc_test = table.iloc[train_idx], table.iloc[test_idx]
        yc_train, yc_test = label[train_idx], label[test_idx]

        # Create and train the Random Forest model with the training fold
        model = RandomForestClassifier(**params)
        model.fit(Xc_train, yc_train)

        # Make predictions on the testing fold
        preds = model.predict(Xc_test)

        # Calculate the accuracy and other metrics of the current fold
        fold_accuracy = accuracy_score(yc_test, preds)
        fold_f1 = f1_score(yc_test, preds, average='weighted')
        fold_recall = recall_score(yc_test, preds, average='weighted')
        fold_precision = precision_score(yc_test, preds, average='weighted')

        accuracy_scores.append(fold_accuracy)
        f1_scores.append(fold_f1)
        recall_scores.append(fold_recall)
        precision_scores.append(fold_precision)

    # Calculate the average of each metric across all folds
    avg_accuracy = np.mean(accuracy_scores)
    avg_f1 = np.mean(f1_scores)
    avg_recall = np.mean(recall_scores)
    avg_precision = np.mean(precision_scores)

    # Print the results
    print(f"Average Accuracy: {avg_accuracy * 100:.2f}%")
    print(f"Average F1-Score: {avg_f1:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")

    return model


rfresult_multiclass_cv(hd_df_dummies_pca, y, param_rf)


def xgbresult_multiclass_gridsearch(table, label):
    print('xgboost - multi-class classification with grid search')

    # Define initial model
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=5, verbosity=0, silent=0)

    # Define parameter grid
    param_grid = {
        'eta': [0.02, 0.04, 0.06, 0.08, 0.1],
        'max_depth': [4, 5, 6, 7],
        'lambda': [1, 2, 5],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.3, 0.5, 1, 1.5],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)

    # Split data into training and testing sets
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(table, label, stratify=label, test_size=0.2,
                                                            random_state=100)

    # Fit GridSearchCV
    grid_search.fit(Xc_train, yc_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Make predictions with the best model
    preds = best_model.predict(Xc_test)

    # Calculate the accuracy
    accuracy = accuracy_score(yc_test, preds)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return best_model
