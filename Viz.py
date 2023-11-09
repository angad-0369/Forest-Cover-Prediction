import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('covtype.csv', index_col=0)

# Class distribution
sns.countplot(data=dataset, x='Cover_Type', palette=["#AFEEEE","#B0C4DE"])

# Correlation
size = 10
data = dataset.iloc[:, :size]
data_corr = data.corr()
threshold = 0.5
corr_list = []
for i in range(size):
    for j in range(i+1, size):
        if ((threshold <= data_corr.iloc[i, j] < 1) or (-threshold >= data_corr.iloc[i, j] > -1)) and abs(data_corr.iloc[i, j]) > 0:
            corr_list.append([data_corr.iloc[i, j], i, j])
s_corr_list = sorted(corr_list, key=lambda x: -abs(x[0]))
for v, i, j in s_corr_list:
    sns.pairplot(data=dataset, hue="Cover_Type", height=6, x_vars=data.columns[i], y_vars=data.columns[j], diag_kind="hist", palette="flare")
    plt.show()

# Heat Map
col_list = dataset.columns
col_list = [col for col in col_list if not col.startswith('Soil')]
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data=dataset[col_list].corr(), square=True, linewidths=1, cmap="Blues", annot=True)
plt.title('Correlation of Variables')
plt.show()

# Horizontal_Distance_To_Hydrology & Vertical_Distance_To_Hydrology with Soil_Type1
sns.scatterplot(data=dataset, x='Horizontal_Distance_To_Hydrology', y='Vertical_Distance_To_Hydrology', hue='Soil_Type1', palette="Blues_r")

# Horizontal_Distance_To_Hydrology & Vertical_Distance_To_Hydrology with Wilderness_Area2
sns.scatterplot(data=dataset, x='Horizontal_Distance_To_Hydrology', y='Vertical_Distance_To_Hydrology', hue='Wilderness_Area2', palette="Blues_r")