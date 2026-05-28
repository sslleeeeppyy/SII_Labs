import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
warnings.filterwarnings('ignore')


# =============================================================
# РУЧНАЯ РЕАЛИЗАЦИЯ ПЕРСЕПТРОНА (CustomPerceptron)
# =============================================================
# Персептрон — простейшая нейронная сеть из одного нейрона.
# Алгоритм обучения:
#   1. Инициализируем веса нулями.
#   2. Для каждого объекта считаем взвешенную сумму признаков + смещение (bias).
#   3. Если предсказание неверное — корректируем веса:
#      w = w + learning_rate * y_true * x
#   Повторяем n_iter раз (эпох).
# =============================================================
class CustomPerceptron:
    def __init__(self, learning_rate=1.0, n_iter=100, random_state=None):
        self.learning_rate = learning_rate  # шаг обучения
        self.n_iter = n_iter                # количество эпох
        self.random_state = random_state
        self.weights = None                 # веса признаков
        self.bias = 0                       # смещение (порог)

    def _sign(self, x):
        """Функция активации: если x >= 0 → класс 1, иначе → класс 0"""
        return 1 if x >= 0 else 0

    def fit(self, X, y):
        """Обучение персептрона"""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            for i in range(n_samples):
                # Взвешенная сумма: w·x + bias
                linear_output = np.dot(self.weights, X[i]) + self.bias
                y_pred = self._sign(linear_output)

                # Правило обновления весов (только при ошибке)
                error = y[i] - y_pred
                self.weights += self.learning_rate * error * X[i]
                self.bias    += self.learning_rate * error
        return self

    def predict(self, X):
        """Предсказание классов"""
        X = np.asarray(X, dtype=float)
        results = []
        for x in X:
            linear_output = np.dot(self.weights, x) + self.bias
            results.append(self._sign(linear_output))
        return np.array(results)

    def predict_proba(self, X):
        """Псевдо-вероятности через сигмоиду (для ROC-кривой)"""
        X = np.asarray(X, dtype=float)
        scores = X.dot(self.weights) + self.bias
        # Сигмоида: преобразует любое число в диапазон (0, 1)
        proba_positive = 1 / (1 + np.exp(-scores))
        return np.column_stack([1 - proba_positive, proba_positive])


# =============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# =============================================================
train = pd.read_csv('disease_train.csv')
test  = pd.read_csv('disease_public_test.csv')
sub   = pd.read_csv('disease_sample_submission.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nПервые 5 строк train:")
print(train.head())
print("\nБаланс классов в train:")
print(train['Y'].value_counts())
print("\nПропуски в train:\n", train.isnull().sum())
print("Пропуски в test:\n", test.isnull().sum())

# Разделяем признаки и целевую переменную
X_train = train.drop('Y', axis=1)
y_train = train['Y']
X_test  = test.copy()
y_test  = sub['Y']


# =============================================================
# 2. ОБУЧЕНИЕ НА СЫРЫХ ДАННЫХ
# =============================================================
print("\n" + "="*60)
print("ОЦЕНКА НА СЫРЫХ ДАННЫХ")
print("="*60)

# --- ExtraTreesClassifier (sklearn) ---
# Строит много деревьев на случайных подвыборках с случайными порогами,
# затем усредняет их предсказания. n_estimators=100 — количество деревьев.
etc_raw = ExtraTreesClassifier(n_estimators=100, random_state=42)
etc_raw.fit(X_train, y_train)
y_pred_etc_raw   = etc_raw.predict(X_test)
y_proba_etc_raw  = etc_raw.predict_proba(X_test)[:, 1]

print("\n--- ExtraTreesClassifier (sklearn, сырые данные) ---")
print(classification_report(y_test, y_pred_etc_raw))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_etc_raw))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba_etc_raw), 4))

# --- Perceptron (sklearn) ---
# Персептрон чувствителен к масштабу, поэтому на сырых данных может плохо работать.
# max_iter=1000 — максимум итераций, tol — порог сходимости.
perc_raw = Perceptron(max_iter=1000, random_state=42, tol=1e-3)
perc_raw.fit(X_train, y_train)
y_pred_perc_raw = perc_raw.predict(X_test)
# Sklearn Perceptron не имеет predict_proba — используем decision_function
y_score_perc_raw = perc_raw.decision_function(X_test)

print("\n--- Perceptron (sklearn, сырые данные) ---")
print(classification_report(y_test, y_pred_perc_raw))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_perc_raw))
print("ROC-AUC:", round(roc_auc_score(y_test, y_score_perc_raw), 4))

# --- CustomPerceptron (ручная реализация, сырые данные) ---
# Данные нужно масштабировать даже здесь, иначе веса не сойдутся —
# но для демонстрации запускаем на сырых.
custom_perc_raw = CustomPerceptron(learning_rate=0.01, n_iter=100, random_state=42)
custom_perc_raw.fit(X_train.values, y_train.values)
y_pred_custom_raw   = custom_perc_raw.predict(X_test.values)
y_proba_custom_raw  = custom_perc_raw.predict_proba(X_test.values)[:, 1]

print("\n--- CustomPerceptron (ручная, сырые данные) ---")
print(classification_report(y_test, y_pred_custom_raw))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_custom_raw))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba_custom_raw), 4))


# =============================================================
# 3. ПРЕДОБРАБОТКА — StandardScaler
# =============================================================
# StandardScaler приводит каждый признак к среднему=0 и std=1.
# Важно: fit только на train, transform на test (чтобы не было утечки данных).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# =============================================================
# 4. ОБУЧЕНИЕ НА МАСШТАБИРОВАННЫХ ДАННЫХ
# =============================================================
print("\n" + "="*60)
print("ОЦЕНКА ПОСЛЕ МАСШТАБИРОВАНИЯ (StandardScaler)")
print("="*60)

# --- ExtraTreesClassifier scaled ---
etc_scaled = ExtraTreesClassifier(n_estimators=100, random_state=42)
etc_scaled.fit(X_train_scaled, y_train)
y_pred_etc_scaled  = etc_scaled.predict(X_test_scaled)
y_proba_etc_scaled = etc_scaled.predict_proba(X_test_scaled)[:, 1]

print("\n--- ExtraTreesClassifier (sklearn, масштабированные) ---")
print(classification_report(y_test, y_pred_etc_scaled))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_etc_scaled))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba_etc_scaled), 4))

# --- Perceptron scaled ---
perc_scaled = Perceptron(max_iter=1000, random_state=42, tol=1e-3)
perc_scaled.fit(X_train_scaled, y_train)
y_pred_perc_scaled  = perc_scaled.predict(X_test_scaled)
y_score_perc_scaled = perc_scaled.decision_function(X_test_scaled)

print("\n--- Perceptron (sklearn, масштабированные) ---")
print(classification_report(y_test, y_pred_perc_scaled))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_perc_scaled))
print("ROC-AUC:", round(roc_auc_score(y_test, y_score_perc_scaled), 4))

# --- CustomPerceptron scaled ---
custom_perc_scaled = CustomPerceptron(learning_rate=0.01, n_iter=100, random_state=42)
custom_perc_scaled.fit(X_train_scaled, y_train.values)
y_pred_custom_scaled  = custom_perc_scaled.predict(X_test_scaled)
y_proba_custom_scaled = custom_perc_scaled.predict_proba(X_test_scaled)[:, 1]

print("\n--- CustomPerceptron (ручная, масштабированные) ---")
print(classification_report(y_test, y_pred_custom_scaled))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_custom_scaled))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba_custom_scaled), 4))


# =============================================================
# 5. СВОДНАЯ ТАБЛИЦА МЕТРИК
# =============================================================
results = pd.DataFrame({
    'Модель': [
        'ExtraTrees raw', 'ExtraTrees scaled',
        'Perceptron raw', 'Perceptron scaled',
        'CustomPerceptron raw', 'CustomPerceptron scaled'
    ],
    'Accuracy': [
        accuracy_score(y_test, y_pred_etc_raw),
        accuracy_score(y_test, y_pred_etc_scaled),
        accuracy_score(y_test, y_pred_perc_raw),
        accuracy_score(y_test, y_pred_perc_scaled),
        accuracy_score(y_test, y_pred_custom_raw),
        accuracy_score(y_test, y_pred_custom_scaled),
    ],
    'ROC-AUC': [
        roc_auc_score(y_test, y_proba_etc_raw),
        roc_auc_score(y_test, y_proba_etc_scaled),
        roc_auc_score(y_test, y_score_perc_raw),
        roc_auc_score(y_test, y_score_perc_scaled),
        roc_auc_score(y_test, y_proba_custom_raw),
        roc_auc_score(y_test, y_proba_custom_scaled),
    ]
})
results['Accuracy'] = results['Accuracy'].round(4)
results['ROC-AUC']  = results['ROC-AUC'].round(4)

print("\n" + "="*60)
print("СВОДНАЯ ТАБЛИЦА МЕТРИК")
print("="*60)
print(results.to_string(index=False))


# =============================================================
# 6. РАЗНЫЕ МЕТОДЫ РАЗБИЕНИЯ ДАННЫХ (на масштабированных)
# =============================================================
print("\n" + "="*60)
print("РАЗНЫЕ МЕТОДЫ ФОРМИРОВАНИЯ TRAIN/TEST (масштабированные)")
print("="*60)

# Объединяем все данные для честного случайного разбиения
all_data = pd.concat([
    train,
    pd.DataFrame(X_test, columns=X_train.columns).assign(Y=y_test.values)
], ignore_index=True)
X_all = all_data.drop('Y', axis=1)
y_all = all_data['Y']

# Способ 1: train_test_split random_state=42
X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
sc1 = StandardScaler()
X_tr1_s = sc1.fit_transform(X_tr1)
X_te1_s  = sc1.transform(X_te1)
m1 = ExtraTreesClassifier(n_estimators=100, random_state=42).fit(X_tr1_s, y_tr1)
print(f"1) train_test_split (test_size=0.3, random_state=42):  accuracy = {accuracy_score(y_te1, m1.predict(X_te1_s)):.4f}")

# Способ 2: train_test_split random_state=123
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_all, y_all, test_size=0.3, random_state=123)
sc2 = StandardScaler()
X_tr2_s = sc2.fit_transform(X_tr2)
X_te2_s  = sc2.transform(X_te2)
m2 = ExtraTreesClassifier(n_estimators=100, random_state=42).fit(X_tr2_s, y_tr2)
print(f"2) train_test_split (test_size=0.3, random_state=123): accuracy = {accuracy_score(y_te2, m2.predict(X_te2_s)):.4f}")

# Способ 3: 5-кратная кросс-валидация
# cross_val_score сам разбивает данные на 5 частей и усредняет результат
cv_scores = cross_val_score(
    ExtraTreesClassifier(n_estimators=100, random_state=42),
    X_train_scaled, y_train, cv=5, scoring='accuracy'
)
print(f"3) 5-fold кросс-валидация (train, масштаб.): accuracy = {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")


# =============================================================
# 7. КЛАСТЕРИЗАЦИЯ — KMeans
# =============================================================
# KMeans — обучение БЕЗ учителя. Не знает меток Y, сам делит на 2 группы.
# Потом сравниваем, совпали ли кластеры с реальными метками.
print("\n" + "="*60)
print("КЛАСТЕРИЗАЦИЯ (KMeans, n_clusters=2)")
print("="*60)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_train_scaled)
test_clusters = kmeans.predict(X_test_scaled)

ari = adjusted_rand_score(y_test, test_clusters)
ami = adjusted_mutual_info_score(y_test, test_clusters)
sil = silhouette_score(X_test_scaled, test_clusters)

print(f"Adjusted Rand Index (ARI):  {ari:.3f}")
print(f"Adjusted Mutual Info (AMI): {ami:.3f}")
print(f"Silhouette Score (test):    {sil:.3f}")


# =============================================================
# 8. ВИЗУАЛИЗАЦИЯ
# =============================================================

# --- График 1: Scatter plot кластеров ---
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1],
            c=test_clusters, cmap='viridis', alpha=0.6)
plt.xlabel('X1 (scaled)'); plt.ylabel('X2 (scaled)')
plt.title('Кластеры KMeans')
plt.colorbar(label='Кластер')

plt.subplot(1, 2, 2)
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1],
            c=y_test, cmap='coolwarm', alpha=0.6)
plt.xlabel('X1 (scaled)'); plt.ylabel('X2 (scaled)')
plt.title('Истинные метки Y')
plt.colorbar(label='Y (0/1)')
plt.tight_layout()
plt.savefig('clusters.png', dpi=100)
plt.show()

# --- График 2: ROC-кривые ---
plt.figure(figsize=(9, 6))
for label, y_score in [
    ('ExtraTrees raw',     y_proba_etc_raw),
    ('ExtraTrees scaled',  y_proba_etc_scaled),
    ('Perceptron raw',     y_score_perc_raw),
    ('Perceptron scaled',  y_score_perc_scaled),
    ('CustomPerc raw',     y_proba_custom_raw),
    ('CustomPerc scaled',  y_proba_custom_scaled),
]:
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)
    style = '--' if 'Custom' in label else '-'
    plt.plot(fpr, tpr, style, label=f'{label} (AUC={auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Случайная модель')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые')
plt.legend(fontsize=8)
plt.grid()
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=100)
plt.show()

# --- График 3: Матрицы ошибок ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
pairs = [
    ('ExtraTrees raw',    y_pred_etc_raw,     axes[0][0]),
    ('ExtraTrees scaled', y_pred_etc_scaled,  axes[0][1]),
    ('Perceptron raw',    y_pred_perc_raw,    axes[1][0]),
    ('Perceptron scaled', y_pred_perc_scaled, axes[1][1]),
]
for title, y_pred, ax in pairs:
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Матрица ошибок: {title}')
    ax.set_xlabel('Предсказано'); ax.set_ylabel('Истина')
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=100)
plt.show()

# --- График 4: Важность признаков ExtraTrees ---
feature_names = X_train.columns.tolist()
importances = etc_scaled.feature_importances_
plt.figure(figsize=(8, 4))
plt.bar(feature_names, importances, color='steelblue')
plt.title('Важность признаков (ExtraTreesClassifier)')
plt.xlabel('Признак'); plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100)
plt.show()

print("\nГотово! Все графики сохранены.")