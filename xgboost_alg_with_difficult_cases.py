import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('ST14000NM001G.csv')

# Преобразование даты в формат datetime
data['date'] = pd.to_datetime(data['date'])

# Сортировка данных по серийному номеру и дате
data = data.sort_values(by=['serial_number', 'date'])

# Создание целевой переменной
def create_target_variable(group):
    if group['failure'].max() == 1:
        failure_date = group[group['failure'] == 1]['date'].min()
        group['will_fail_in_30_days'] = (failure_date - group['date']).dt.days <= 30
    else:
        group['will_fail_in_30_days'] = 0
    return group

data = data.groupby('serial_number').apply(create_target_variable)

# Выбор признаков
features_to_normalize = ['smart_5_raw', 'smart_9_raw', 'smart_187_raw', 'smart_188_raw',
                         'smart_192_raw', 'smart_197_raw', 'smart_198_raw', 'smart_199_raw',
                         'smart_240_raw', 'smart_241_raw', 'smart_242_raw']

# Нормализация признаков
scaler = StandardScaler()
data[features_to_normalize] = scaler.fit_transform(data[features_to_normalize])

# Разделение данных на признаки и целевую переменную
X = data.drop(['date', 'serial_number', 'model', 'capacity_bytes', 'failure', 'will_fail_in_30_days'], axis=1)
y = data['will_fail_in_30_days']

# Разделение данных на тренировочную и две тестовые выборки
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_test_1, X_test_2, y_test_1, y_test_2 = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Применение SMOTE для балансировки классов на тренировочной выборке
smote = SMOTE(sampling_strategy=0.1, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Увеличение важности ключевых признаков
important_features = ['smart_187_raw', 'smart_188_raw', 'smart_198_raw']
weight_factor = 3  # Количество копий признаков для увеличения их влияния

# Добавление копий ключевых признаков
for feature in important_features:
    for i in range(weight_factor - 1):  # -1, так как оригинальный признак уже существует
        X_train_resampled[f'{feature}_copy_{i+1}'] = X_train_resampled[feature]
        X_test_1[f'{feature}_copy_{i+1}'] = X_test_1[feature]
        X_test_2[f'{feature}_copy_{i+1}'] = X_test_2[feature]


# Модель XGBoost
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    learning_rate=0.1,
    n_estimators=300,
    max_depth=15,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    scale_pos_weight=20,
    random_state=42,
    n_jobs=-1,
)

# Первичное обучение модели
model.fit(X_train_resampled, y_train_resampled)

# Проверка на первой тестовой выборке
y_pred_prob_1 = model.predict_proba(X_test_1)[:, 1]
y_pred_1 = (y_pred_prob_1 >= 0.4).astype(int)

# Выделение сложных примеров из первой тестовой выборки
test_predictions_1 = X_test_1.copy()
test_predictions_1['actual'] = y_test_1
test_predictions_1['predicted'] = y_pred_1
test_predictions_1['probability'] = y_pred_prob_1

# Ошибки модели
false_positives_1 = test_predictions_1[(test_predictions_1['actual'] == 0) & (test_predictions_1['predicted'] == 1)]
false_negatives_1 = test_predictions_1[(test_predictions_1['actual'] == 1) & (test_predictions_1['predicted'] == 0)]

# Соединение сложных примеров
difficult_cases_1 = pd.concat([false_positives_1, false_negatives_1])

# Увеличение сложных примеров с помощью SMOTE
smote_difficult = SMOTE(sampling_strategy=1, random_state=42)
X_difficult_resampled_1, y_difficult_resampled_1 = smote_difficult.fit_resample(
    difficult_cases_1.drop(['actual', 'predicted', 'probability'], axis=1),
    difficult_cases_1['actual']
)

for i in range(3):
    X_difficult_resampled_1 = pd.concat([X_difficult_resampled_1, X_difficult_resampled_1])
    y_difficult_resampled_1 = pd.concat([y_difficult_resampled_1, y_difficult_resampled_1])


# # Объединение с основным тренировочным набором
X_train_final = pd.concat([X_train_resampled, X_difficult_resampled_1])
y_train_final = pd.concat([y_train_resampled, y_difficult_resampled_1])

# Повторное обучение модели
model.fit(X_train_final, y_train_final)

# Проверка на второй тестовой выборке
y_pred_prob_2 = model.predict_proba(X_test_2)[:, 1]
y_pred_2 = (y_pred_prob_2 >= 0.7).astype(int)

# Оценка модели на второй тестовой выборке
accuracy_2 = accuracy_score(y_test_2, y_pred_2)
roc_auc_2 = roc_auc_score(y_test_2, y_pred_prob_2)
f1_2 = f1_score(y_test_2, y_pred_2)
cm_2 = confusion_matrix(y_test_2, y_pred_2)

# Вывод результатов
print(f'Accuracy (Test 2): {accuracy_2:.4f}')
print(f'AUC-ROC (Test 2): {roc_auc_2:.4f}')
print(f'F1-Score (Test 2): {f1_2:.4f}')
print('Classification Report (Test 2):')
print(classification_report(y_test_2, y_pred_2))
print('Confusion Matrix (Test 2):')
print(cm_2)

# Визуализация ROC-кривой для второй тестовой выборки
# fpr_2, tpr_2, _ = roc_curve(y_test_2, y_pred_prob_2)
# plt.plot(fpr_2, tpr_2, label=f'ROC curve (area = {roc_auc_2:.2f})')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve (Test 2)')
# plt.legend()
# plt.show()

# Сохранение результатов
# test_predictions_2 = X_test_2.copy()
# test_predictions_2['actual'] = y_test_2
# test_predictions_2['final_predicted'] = y_pred_2
# test_predictions_2['final_probability'] = y_pred_prob_2
# test_predictions_2.to_csv('test_predictions_with_difficult_cases_test2.csv', index=False)

# Сохранение модели
# joblib.dump(model, 'xgboost_model_with_difficult_cases_test2.pkl')
