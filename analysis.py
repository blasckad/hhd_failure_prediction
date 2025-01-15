import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

# Загрузка данных
data = pd.read_csv('ST14000NM001G.csv')

# Преобразование даты в формат datetime
data['date'] = pd.to_datetime(data['date'])

# Сортировка данных по серийному номеру и дате
data = data.sort_values(by=['serial_number', 'date'])

# Анализ пропусков
missing_values = data.isnull().sum()
print("Missing values:")
print(missing_values[missing_values > 0])

# Выбор SMART-показателей для анализа
smart_features = ['smart_5_raw', 'smart_9_raw', 'smart_187_raw', 'smart_188_raw',
                  'smart_192_raw', 'smart_197_raw', 'smart_198_raw', 'smart_199_raw',
                  'smart_240_raw', 'smart_241_raw', 'smart_242_raw']

# Описательная статистика
print("Descriptive statistics for SMART features:")
print(data[smart_features].describe())

# Корреляционный анализ
correlation_matrix = data[smart_features].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of SMART Features')
plt.show()

# Временные ряды для нескольких дисков
sample_disks = data['serial_number'].drop_duplicates().sample(20, random_state=42) # Возьмем 20 дисков
sample_data = data[data['serial_number'].isin(sample_disks)]

plt.figure(figsize=(20, 20))
for i, feature in enumerate(smart_features):
    plt.subplot(5, 4, i + 1)
    for serial_number in sample_disks:
        disk_data = sample_data[sample_data['serial_number'] == serial_number]
        plt.plot(disk_data['date'], disk_data[feature])
    plt.title(f"Time Series of {feature}")
    # plt.xlabel("Date")
    plt.ylabel(feature)
    # plt.legend()
    plt.gca().xaxis.set_visible(False)
plt.tight_layout()
plt.show()

# Загрузка и подготовка данных
data = pd.read_csv('ST14000NM001G.csv')


# Предобработка
data['date'] = pd.to_datetime(data['date'])
smart_features = ['smart_5_raw', 'smart_9_raw', 'smart_187_raw',
                  'smart_188_raw', 'smart_192_raw', 'smart_197_raw',
                  'smart_198_raw', 'smart_199_raw', 'smart_240_raw',
                  'smart_241_raw', 'smart_242_raw']

# Определение корреляции с поломкой
failure_corr = data[smart_features + ['failure']].corr()['failure'].sort_values(ascending=False)
print(failure_corr)

# Визуализация корреляций
sns.barplot(x=failure_corr.index, y=failure_corr.values)
plt.xticks(rotation=45)
plt.title("Корреляция показателей SMART с поломкой")
plt.show()

# Разделение данных: сломанные и исправные диски
failure_data = data[data['failure'] == 1]
healthy_data = data[data['failure'] == 0]

# Анализ распределений для каждого параметра
critical_thresholds = {}
for feature in smart_features:
    plt.figure(figsize=(10, 5))
    sns.histplot(healthy_data[feature], label="Без поломки", color="green", kde=True, stat="density", bins=30)
    sns.histplot(failure_data[feature], label="Перед поломкой", color="red", kde=True, stat="density", bins=30)
    plt.title(f"Распределение значений {feature}")
    plt.legend()
    plt.show()

