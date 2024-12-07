import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Загрузка и подготовка данных
data = pd.read_csv('ST14000NM001G.csv')

data['date'] = pd.to_datetime(data['date'])

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

# Нормализация признаков
features_to_normalize = ['smart_5_raw', 'smart_9_raw', 'smart_187_raw', 'smart_188_raw',
                         'smart_192_raw', 'smart_197_raw', 'smart_198_raw', 'smart_199_raw',
                         'smart_240_raw', 'smart_241_raw', 'smart_242_raw']

scaler = StandardScaler()
data[features_to_normalize] = scaler.fit_transform(data[features_to_normalize])

# Разделение данных на признаки и целевую переменную
X = data.drop(['date', 'serial_number', 'model', 'capacity_bytes', 'failure', 'will_fail_in_30_days'], axis=1)
y = data['will_fail_in_30_days']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=80, stratify=y)

# Балансировка
smote = SMOTE(sampling_strategy=0.1, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Увеличение важности критических признаков
important_features = ['smart_187_raw', 'smart_188_raw', 'smart_198_raw']

weight_factor = 50
X_train_resampled[important_features] *= weight_factor
X_test[important_features] *= weight_factor


# Настройка Модели
model = xgb.XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss', 
    learning_rate=0.1,
    n_estimators=300,
    max_depth=10,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    scale_pos_weight=20,
    random_state=42,
    n_jobs=-1,
)

# Обучение модели
model.fit(X_train_resampled, y_train_resampled)

# Сохранение модели
joblib.dump(model, 'xgboost_model_big.pkl')

# Тестирование
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob >= 0.5).astype(int)  # Применение порога 0.4 для классификации

# Сохранение тестовых данных
test_predictions = X_test.copy()
test_predictions['will_fail_in_30_days_actual'] = y_test
test_predictions['will_fail_in_30_days_predicted'] = y_pred
test_predictions['prediction_probability'] = y_pred_prob

test_predictions.to_csv('test_predictions.csv', index=False)

# Оценка модели
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Вывод метрик
print(f'Accuracy: {accuracy:.4f}')
print(f'AUC-ROC: {roc_auc:.4f}')
print(f'F1-Score: {f1:.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(cm)

# Визуализация ROC-кривой
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Визуализация важности признаков
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.show()