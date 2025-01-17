import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import ADASYN
import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('ST14000NM001G.csv')

# Преобразование даты
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(by=['serial_number', 'date'])

# Создание целевой переменной
def create_target_variable(group):
    if group['failure'].max() == 1:
        failure_date = group[group['failure'] == 1]['date'].min()
        failure_days = (failure_date - pd.Timestamp("1970-01-01")).days
        group_days = (group['date'] - pd.Timestamp("1970-01-01")).dt.days
        group['will_fail_in_30_days'] = (failure_days - group_days <= 31).astype(int)
    else:
        group['will_fail_in_30_days'] = 0
        return group.iloc[:-30] # Удаление последних 30 дней для дисков без поломок
    return group

# Применение функции для создания целевой переменной
data = data.groupby('serial_number').apply(create_target_variable)

def add_features(group):
    # Логарифмические признаки
    group['smart_9_raw_log'] = np.log1p(group['smart_9_raw'])
    group['smart_241_raw_log'] = np.log1p(group['smart_241_raw'])
    group['smart_242_raw_log'] = np.log1p(group['smart_242_raw'])

    # Признаки, связанные с износом диска
    group['smart_241_raw_per_hour'] = group['smart_241_raw'] / group['smart_9_raw'].replace(0, 1)
    group['smart_242_raw_per_hour'] = group['smart_242_raw'] / group['smart_9_raw'].replace(0, 1)

    # Признаки, связанные с ошибками
    group['smart_5_raw_diff'] = group['smart_5_raw'].diff().fillna(0)
    group['smart_197_raw_diff'] = group['smart_197_raw'].diff().fillna(0)
    group['smart_198_raw_diff'] = group['smart_198_raw'].diff().fillna(0)
    group['smart_187_raw_diff'] = group['smart_187_raw'].diff().fillna(0)
    group['smart_188_raw_diff'] = group['smart_188_raw'].diff().fillna(0)

    # Признаки, связанные с нагрузкой на диск
    group['smart_241_242_ratio'] = group['smart_241_raw'] / group['smart_242_raw'].replace(0, 1)
    group['smart_240_9_ratio'] = group['smart_240_raw'] / group['smart_9_raw'].replace(0, 1)

    # Признаки, связанные с состоянием диска
    group['smart_5_197_ratio'] = group['smart_5_raw'] / group['smart_197_raw'].replace(0, 1)
    group['smart_197_198_ratio'] = group['smart_197_raw'] / group['smart_198_raw'].replace(0, 1)

    # Признаки, связанные с ошибками
    group['smart_199_diff'] = group['smart_199_raw'].diff().fillna(0)
    group['smart_199_per_hour'] = group['smart_199_raw'] / group['smart_9_raw'].replace(0, 1)

    # Взаимодействие признаков
    group['smart_187_smart_197'] = group['smart_187_raw'] * group['smart_197_raw']
    group['smart_5_smart_197'] = group['smart_5_raw'] * group['smart_197_raw']
    group['smart_197_smart_198'] = group['smart_197_raw'] * group['smart_198_raw']
    group['smart_241_smart_242'] = group['smart_241_raw'] * group['smart_242_raw']
    group['smart_9_smart_240'] = group['smart_9_raw'] * group['smart_240_raw']

    return group

# Сброс индекса перед группировкой
data = data.reset_index(drop=True)

# Применение функции для добавления признаков
data = data.groupby('serial_number').apply(add_features)

# Убираем пропуски, если они есть
data = data.dropna()

# Нормализация признаков
features_to_normalize = [
    'smart_5_raw', 'smart_9_raw', 'smart_187_raw', 'smart_188_raw',
    'smart_192_raw', 'smart_197_raw', 'smart_198_raw', 'smart_199_raw',
    'smart_240_raw', 'smart_241_raw', 'smart_242_raw', "smart_9_raw_log",
    "smart_241_raw_log", "smart_242_raw_log", "smart_241_raw_per_hour",
    "smart_242_raw_per_hour", "smart_5_raw_diff", "smart_197_raw_diff",
    "smart_198_raw_diff", "smart_187_raw_diff", "smart_188_raw_diff",
    "smart_241_242_ratio", "smart_240_9_ratio", "smart_5_197_ratio",
    "smart_197_198_ratio", "smart_199_diff", "smart_199_per_hour",
    "smart_187_smart_197", "smart_5_smart_197", "smart_197_smart_198",
    "smart_241_smart_242", "smart_9_smart_240"
]


scaler = MinMaxScaler()
data[features_to_normalize] = scaler.fit_transform(data[features_to_normalize])

# Разделение данных на признаки и целевую переменную
X = data.drop(['date', 'serial_number', 'model', 'capacity_bytes', 'failure', 'will_fail_in_30_days'], axis=1)
y = data['will_fail_in_30_days']

# Разделение данных на обучающую, валидационную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Применение ADASYN для балансировки классов
adasyn = ADASYN(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)

# Определение весов признаков
important_features = ['smart_5_raw', 'smart_187_raw',
                      'smart_192_raw', 'smart_197_raw', 'smart_198_raw']
feature_weights = [5.0 if feature in important_features else 1.0 for feature in X.columns]

# Определение весов для примеров: больший вес для примеров из класса 1
weights_train = np.where(y_train_resampled == 1, 5.0, 1.0)  # Вес = 5 для поломок, вес = 1 для исправных дисков

# Создание DMatrix с весами для примеров
dtrain = xgb.DMatrix(X_train_resampled, label=y_train_resampled, weight=weights_train, feature_weights=feature_weights)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# Параметры модели с использованием GPU
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 9,  # Максимальная глубина дерева
    'eta': 0.1,  # Скорость обучения
    'subsample': 1,  # Доля примеров для обучения каждого дерева
    'colsample_bytree': 1,  # Доля признаков для обучения каждого дерева
    'seed': 42,
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor'
}

# Обучение модели с использованием валидационной выборки и ранней остановки
evals = [(dtrain, 'train'), (dval, 'val')]  # Данные для оценки
model = xgb.train(
    params,
    dtrain,
    num_boost_round=5000,  # Максимальное количество деревьев
    evals=evals,  # Данные для оценки
    early_stopping_rounds=100,  # Остановка, если качество не улучшается 100 раундов
    verbose_eval=100,  # Вывод метрики каждые 100 раундов
)

# Функция для оценки модели
def evaluate_model(threshold):
    y_pred_prob = model.predict(dtest)
    y_pred = (y_pred_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f'\n--- Evaluation with Threshold: {threshold} ---')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'AUC-ROC: {roc_auc:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix:')
    print(cm)


# Первоначальная оценка
evaluate_model(0.8)

# Ввод порога
while True:
    try:
        threshold = float(input("\nВведите порог классификации: "))
        evaluate_model(threshold)
    except ValueError:
        print("Выход.")
        break

def plot_roc_curve(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def plot_precision_recall_curve(y_true, y_pred_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

# Визуализация ROC-кривой и Precision-Recall кривой
y_pred_prob = model.predict(dtest)
plot_roc_curve(y_test, y_pred_prob)
plot_precision_recall_curve(y_test, y_pred_prob)