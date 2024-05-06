"""Dataset column descriptions

ID: Patient identifier (int)
Age: Age of the patient (in years)
Schooling: time the patient spent in school (in years)
Gender: 1=male, 2=female
Breastfeeding: 1=yes, 2=no, 3=unknown
Varicella: 1=positive, 2=negative, 3=unknown
Initial_Symptoms: 1=visual, 2=sensory, 3=motor, 4=other, 5= visual and sensory, 6=visual and motor, 7=visual and others, 8=sensory and motor, 9=sensory and other, 10=motor and other, 11=Visual, sensory and motor, 12=visual, sensory and other, 13=Visual, motor and other, 14=Sensory, motor and other, 15=visual,sensory,motor and other
Mono _or_Polysymptomatic: 1=monosymptomatic, 2=polysymptomatic, 3=unknown
Oligoclonal_Bands: 0=negative, 1=positive, 2=unknown
LLSSEP: 0=negative, 1=positive
ULSSEP:0=negative, 1=positive
VEP:0=negative, 1=positive
BAEP: 0=negative, 1=positive
Periventricular_MRI:0=negative, 1=positive
Cortical_MRI: 0=negative, 1=positive
Infratentorial_MRI:0=negative, 1=positive
Spinal_Cord_MRI: 0=negative, 1=positive
initial_EDSS:?
final_EDSS:?
Group: 1=CDMS, 2=non-CDMS
Definition of some of the technical/medical terms [ref. from wikipedia if not stated explicitly].
Varicella : Another name for Chickenpox, or chicken pox, is a highly contagious disease caused by the initial infection with varicella zoster virus (VZV), a member of the herpesvirus family.
BAEP: In human neuroanatomy, brainstem auditory evoked potentials (BAEPs), also called brainstem auditory evoked responses (BAERs), are very small auditory evoked potentials in response to an auditory stimulus, which are recorded by electrodes placed on the scalp.
VEP: Visual evoked potential (VEP) is an evoked potential elicited by presenting light flash or pattern stimulus which can be used to confirm damage to visual pathway including retina, optic nerve, optic chiasm, optic radiations, and occipital cortex.
Oligoclonal bands: Oligoclonal bands (OCBs) are bands of immunoglobulins that are seen when a patient’s blood serum, or cerebrospinal fluid (CSF) is analyzed. They are used in the diagnosis of various neurological and blood diseases. Oligoclonal bands are present in the CSF of more than 95% of patients with clinically definite multiple sclerosis.
SSEP : Somatosensory evoked potentials (SSEP) are recorded from the central nervous system following stimulation of peripheral nerves. ULSSEP (upper limb SSEP), LLSSEP (lower limb SSEP)
EDSS: The Expanded Disability Status Scale (EDSS) is a method of quantifying disability in multiple sclerosis and monitoring changes in the level of disability over time. It is widely used in clinical trials and in the assessment of people with MS. 2"""

import os.path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

df = pd.read_csv("../Data/conversion_predictors_of_clinically_isolated_syndrome_to_multiple_sclerosis.csv", index_col=0)
# I notice there's a column 0 "Unnamed" which is the actual index, so I added index_col=0.
# a glimpse to the db
print(df.head(2))
# as a rule of thumb, I drop duplicates, should any be in the dataframe to make it slimmer.
df.drop_duplicates(inplace=True)
# I want to see the raw amount of data I have and the type of data
print(df.shape)
# 273 entries, all numerical => I can do np operation with them.
print(df.info())
print("TOTAL NAN: ", df.isnull().sum())
# I notice there are 148 entries missing to two columns: Initial and Final EDSS
nans = df.isnull().sum().sort_values(ascending=False)

if not os.path.exists("../plots/multiple_sclerosis_nans.png"):
    plt.barh(nans.index, nans.values)
    plt.xlabel('Number of Missing Values')
    plt.ylabel('Columns')
    plt.title('Missing Values per Column')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig("../plots/multiple_sclerosis_nans.png")
    plt.show()

print(df.columns)

# lets explore the diversity of the dataset


# this dataset is small, mostly categorical with a few descriptive features
# from a look at the summary I could convert all the dtype to int and keep the nans
# to do so, I'll create a fourth category to EDSS, substitute 0 to Initial symptom and
# use the average distribution in schooling
mean_schooling = int(df['Schooling'].mean())
df['Schooling'] = np.where(df['Schooling'].isnull(), mean_schooling, df['Schooling']).astype(int)
df['Initial_Symptom'] = np.where(df['Initial_Symptom'].isnull(), 0, df['Initial_Symptom']).astype(int)
df['group'] = np.where(df['group'] == 1, 0, df['group'])
df['group'] = np.where(df['group'] == 2, 1, df['group'])

for column_edss in df.columns:
    if "EDSS" in column_edss:
        df[column_edss] = np.where(df[column_edss].isnull(), 0, df[column_edss]).astype(int)

for colum in df.columns:
    uniques = df[colum].unique().tolist()
    uniques.sort()
    print(colum, ": ", uniques)

# now the dataframe has no nan, is type-consistent and we can mold it
print("Let's describe the df:\n")
print(df.describe())
df_num_corr = df.corr()['group'][:-1]
# Plot hd_num_corr
if not os.path.exists('../plots/MS_correlation.png'):
    plt.figure(figsize=(10, 6))
    df_num_corr.plot(kind='bar', color='skyblue')
    plt.title('Correlation with Group Class')
    plt.xlabel('Numerical Features')
    plt.ylabel('Correlation Coefficient')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('../plots/MS_correlation.png')
    plt.show()
top_features = df_num_corr[abs(df_num_corr) > 0.5].sort_values(
    ascending=False)  # displays pearsons correlation coefficient greater than 0.5
print("There is {} strongly correlated values with pre_class:\n{}".format(len(top_features), top_features))

# the Pearson Coefficient showed an inverse linear relationship with the Periventricular_MRI.
# I am doubtful about the EDSS as they contain mostly Unknown values that I set to 0
# intriguingly the negative correlation coeff for periventricular_MRI is counterintuitive:
# the database could be biased or the scan could be linked to other cerbrovascular diseases.

# let's explore the description of age and diagnosis of the database
# let's plot the incidence by age

bins = pd.interval_range(int(df['Age'].min()), int(df['Age'].max() + 1), freq=5)
labels = [f"{x.left}-{x.right}" for x in bins]
df['Age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
df['group'] = df['group'].apply(lambda x: 'MS' if x > 0 else 'No MS')

if not os.path.exists("../plots/MS_by_age.png"):
    # making a new dataframe grouping by age and diagnosis
    age_group_counts = df.groupby(['Age_group', 'group'], observed=False).size().unstack(fill_value=0)
    age_group_counts['Total'] = age_group_counts.sum(axis=1)
    age_group_counts['MS Disease Ratio'] = ((age_group_counts['MS'] / age_group_counts['Total']) * 100).round(2)

    ax = age_group_counts[['No MS', 'MS']].plot(kind='bar', stacked=True, figsize=(10, 6))
    # Annotate each bar with the 'MS Disease Ratio' percentage
    for i in range(len(age_group_counts.index)):
        plt.text(i, age_group_counts.iloc[i]['MS'] + 1, f"{age_group_counts.iloc[i]['MS Disease Ratio']}%", ha='center')
    plt.title('MS Distribution by Age')
    plt.xlabel('Age Group')
    plt.ylabel('Population Number')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Diagnosis', loc='upper left')
    plt.tight_layout()
    plt.savefig("../plots/MS_by_age.png")
    plt.show()

# let's see the distribution between age groups and VEP diagnosis:
if not os.path.exists("../plots/VEP_by_age.png"):
    age_group_counts = df.groupby(['Age_group', 'VEP'], observed=False).size().unstack(fill_value=0)
    age_group_counts['Total'] = age_group_counts.sum(axis=1)
    age_group_counts['VEP+'] = age_group_counts[1]
    age_group_counts['VEP-'] = age_group_counts[0]
    ax = age_group_counts[["VEP+", "VEP-"]].plot(kind='bar', stacked=True, figsize=(10, 6))
    age_group_counts['VEP+ ratio'] = ((age_group_counts[1] / age_group_counts['Total']) * 100).round(2)
    age_group_counts['VEP- ratio'] = ((age_group_counts[0] / age_group_counts['Total']) * 100).round(2)
    # Annotate each bar with the 'MS Disease Ratio' percentage
    for i in range(len(age_group_counts.index)):
        plt.text(i, age_group_counts.iloc[i]['VEP+'] + 1, f"{age_group_counts.iloc[i]['VEP+ ratio']}%", ha='left')
    plt.title('VEP Distribution by Age')
    plt.xlabel('Age Group')
    plt.ylabel('Population Number')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Diagnosis', loc='upper left')
    plt.tight_layout()
    plt.savefig("../plots/VEP_by_age.png")
    plt.show()

# let's evaluate the OCBs and age
if not os.path.exists("../plots/OCB_by_age.png"):
    age_group_counts = df.groupby(['Age_group', 'Oligoclonal_Bands'], observed=False).size().unstack(fill_value=0)
    age_group_counts['Total'] = age_group_counts.sum(axis=1)
    age_group_counts['OCB+'] = age_group_counts[1]
    age_group_counts['OCB-'] = age_group_counts[0]
    ax = age_group_counts[["OCB+", "OCB-"]].plot(kind='bar', stacked=True, figsize=(10, 6))
    age_group_counts['OCB+ ratio'] = ((age_group_counts[1] / age_group_counts['Total']) * 100).round(2)
    age_group_counts['OCB- ratio'] = ((age_group_counts[0] / age_group_counts['Total']) * 100).round(2)
    # Annotate each bar with the 'MS Disease Ratio' percentage
    for i in range(len(age_group_counts.index)):
        plt.text(i, age_group_counts.iloc[i]['OCB+'] + 1, f"{age_group_counts.iloc[i]['OCB+ ratio']}%", ha='left')
    plt.title('OCB Distribution by Age')
    plt.xlabel('Age Group')
    plt.ylabel('Population Number')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Diagnosis', loc='upper left')
    plt.tight_layout()
    plt.savefig("../plots/OCB_by_age.png")
    plt.show()

# statistical analysis

# now lets group every test and see if the positive correlation for each is correlated with the diagnosis groupings
df['tests'] = df[['VEP', 'BAEP', 'Periventricular_MRI', 'Cortical_MRI', 'Infratentorial_MRI',
                  'Spinal_Cord_MRI', 'Oligoclonal_Bands', 'LLSSEP', 'ULSSEP']].any(axis=1)

# AIM: we want to highlight if multiple tests are an accurate predictive measure for diagnosis
contingency_table = pd.crosstab(df['tests'], df['group'])
print("Contingency Table:")
print(contingency_table)
chi2_all, p_value_all, dof_all, expected_all = chi2_contingency(contingency_table)
print("Chi-squared Test statistic:", chi2_all, "P-value", p_value_all)

if not os.path.exists("../plots/CHI_SQUARED_TESTS_COMPARISON.png"):
    # plotting the graph to compare a single test with the overall contingency (can be done for any test)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    contingency_table_vep = pd.crosstab(df['VEP'], df['group'])
    contingency_table_vep.plot(kind='bar', stacked=True, ax=axes[0])
    axes[0].set_title('Contingency Table (VEP)')
    axes[0].set_xlabel('VEP')
    axes[0].set_ylabel('Count')
    axes[0].grid(linestyle='--')
    # let's compare the correlation with the totality of the tests
    contingency_table_all = pd.crosstab(df['tests'], df['group'])
    contingency_table_all.plot(kind='bar', stacked=True, ax=axes[1])
    axes[1].set_title('Contingency Table (All Tests)')
    axes[1].set_xlabel('Tests')
    axes[1].set_ylabel('Count')
    axes[1].grid(linestyle='--')

    # Mostra il plot
    plt.tight_layout()
    plt.savefig("../plots/CHI_SQUARED_TESTS_COMPARISON.png")
    plt.show()
    plt.close()

# the high the Chi-square value is greater than or equal to the critical value
# There is a significant difference between the groups we are studying.
# That is, the difference between actual data and the expected data
# (that assumes the groups aren’t different) is probably too great
# to be attributed to chance. So the multiple tests and the diagnosis are not a coincidence.
# with a p-value < 0.001 the statistical significance is strong.

# ANOVA test across all groups
anova_result = f_oneway(*[group['tests'].values for name, group in df.groupby('group')])
print(f'\n\nANOVA result: F-statistic = {anova_result.statistic:.2f}, p-value = {anova_result.pvalue:.4f}')

# Conducting post-hoc analysis if ANOVA is significant
if anova_result.pvalue < 0.05:
    print("Significant differences were found, performing post-hoc testing...")
    # Performing Tukey's HSD test
    tukey_result = pairwise_tukeyhsd(endog=df['tests'],
                                     groups=df['group'],
                                     alpha=0.05)
    print(tukey_result)
else:
    print("No significant differences were found among groups.")
