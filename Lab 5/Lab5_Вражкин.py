import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from io import StringIO
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ===========================================================
# 1. СИНТЕТИЧЕСКИЙ ДАТАСЕТ (train.tsv / test.tsv)
# ===========================================================
print("=" * 70)
print("1. СИНТЕТИЧЕСКИЙ ДАТАСЕТ (100 признаков)")
print("=" * 70)

train_syn = pd.read_csv('train.tsv', sep='\t', header=None)
test_syn  = pd.read_csv('test.tsv',  sep='\t', header=None)
print(f"Train shape: {train_syn.shape}, Test shape: {test_syn.shape}")

X_syn_full = train_syn.iloc[:, :-1].values
y_syn_full = train_syn.iloc[:, -1].values

X_syn_train, X_syn_val, y_syn_train, y_syn_val = train_test_split(
    X_syn_full, y_syn_full, test_size=0.2, random_state=42
)

scaler_syn = StandardScaler()
X_syn_train_sc = scaler_syn.fit_transform(X_syn_train)
X_syn_val_sc   = scaler_syn.transform(X_syn_val)

# --- ExtraTreesRegressor ---
start = time.perf_counter()
etr_syn = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
etr_syn.fit(X_syn_train_sc, y_syn_train)
y_pred_etr_syn = etr_syn.predict(X_syn_val_sc)
time_etr_syn = time.perf_counter() - start

# --- LinearRegression (второй метод) ---
start = time.perf_counter()
lr_syn = LinearRegression()
lr_syn.fit(X_syn_train_sc, y_syn_train)
y_pred_lr_syn = lr_syn.predict(X_syn_val_sc)
time_lr_syn = time.perf_counter() - start


def print_metrics(y_true, y_pred, name, elapsed=None):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    t_str = f", Время={elapsed:.4f} с" if elapsed is not None else ""
    print(f"{name}: R²={r2:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}{t_str}")


print("\n=== Синтетический датасет (сырые данные) ===")
print_metrics(y_syn_val, y_pred_etr_syn, "ExtraTreesRegressor", time_etr_syn)
print_metrics(y_syn_val, y_pred_lr_syn,  "LinearRegression",    time_lr_syn)

# Кросс-валидация на синтетическом датасете
cv_etr = cross_val_score(ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                         X_syn_train_sc, y_syn_train, cv=5, scoring='r2')
cv_lr  = cross_val_score(LinearRegression(),
                         X_syn_train_sc, y_syn_train, cv=5, scoring='r2')
print(f"\nКросс-валидация (5-fold, R²):")
print(f"  ExtraTrees: mean={cv_etr.mean():.4f}, std={cv_etr.std():.4f}")
print(f"  LinearReg:  mean={cv_lr.mean():.4f},  std={cv_lr.std():.4f}")

# Кластеризация синтетического датасета
kmeans_syn   = KMeans(n_clusters=2, random_state=42)
clusters_syn = kmeans_syn.fit_predict(X_syn_val_sc)
sil_syn      = silhouette_score(X_syn_val_sc, clusters_syn)
median_syn   = np.median(y_syn_val)
y_syn_bin    = (y_syn_val > median_syn).astype(int)
ari_syn      = adjusted_rand_score(y_syn_bin, clusters_syn)
ami_syn      = adjusted_mutual_info_score(y_syn_bin, clusters_syn)
print(f"\nKMeans (синт.): Silhouette={sil_syn:.4f}, ARI={ari_syn:.4f}, AMI={ami_syn:.4f}")


# ===========================================================
# 2. РЕАЛЬНЫЙ ДАТАСЕТ (ml_moscow_flats.csv)
# ===========================================================
print("\n" + "=" * 70)
print("2. РЕАЛЬНЫЙ ДАТАСЕТ — Цены на квартиры в Москве")
print("=" * 70)

# Файл упакован как xlsx, но содержит CSV внутри
df_raw = pd.read_excel('ml_moscow_flats.csv')
col    = df_raw.columns[0]
csv_text = col + '\n' + '\n'.join(df_raw.iloc[:, 0].tolist())
df = pd.read_csv(StringIO(csv_text))

print(f"Размер датасета: {df.shape}")
print(f"Первые 3 строки:\n{df.head(3)}\n")

# Удаляем пропуски
before = len(df)
df = df.dropna()
print(f"Удалено строк с пропусками: {before - len(df)}, осталось: {len(df)}")

# --- Признаки и цель ---
target = 'price'
X = df.drop(target, axis=1)
y = df[target]

numeric_features     = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
print(f"\nЧисловые признаки: {numeric_features}")
print(f"Категориальные признаки: {categorical_features}")

# Предобработка
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- ExtraTreesRegressor Pipeline ---
etr_pipe = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])
start = time.perf_counter()
etr_pipe.fit(X_train, y_train)
y_pred_etr_real = etr_pipe.predict(X_val)
time_etr_real = time.perf_counter() - start

# --- LinearRegression Pipeline ---
lr_pipe = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', LinearRegression())
])
start = time.perf_counter()
lr_pipe.fit(X_train, y_train)
y_pred_lr_real = lr_pipe.predict(X_val)
time_lr_real = time.perf_counter() - start

print("\n=== Реальный датасет (сырые данные) ===")
print_metrics(y_val, y_pred_etr_real, "ExtraTreesRegressor", time_etr_real)
print_metrics(y_val, y_pred_lr_real,  "LinearRegression",    time_lr_real)

# Кросс-валидация на реальном датасете (5-fold)
cv_etr_real = cross_val_score(etr_pipe, X, y, cv=5, scoring='r2')
cv_lr_real  = cross_val_score(lr_pipe,  X, y, cv=5, scoring='r2')
print(f"\nКросс-валидация (5-fold, R²):")
print(f"  ExtraTrees: mean={cv_etr_real.mean():.4f}, std={cv_etr_real.std():.4f}")
print(f"  LinearReg:  mean={cv_lr_real.mean():.4f},  std={cv_lr_real.std():.4f}")

# Кластеризация на реальном датасете
X_all        = preprocessor.fit_transform(X)
kmeans_real  = KMeans(n_clusters=2, random_state=42)
clusters_real = kmeans_real.fit_predict(X_all)
sil_real     = silhouette_score(X_all, clusters_real)
median_real  = np.median(y)
y_bin_real   = (y > median_real).astype(int)
ari_real     = adjusted_rand_score(y_bin_real, clusters_real)
ami_real     = adjusted_mutual_info_score(y_bin_real, clusters_real)
print(f"\nKMeans (реальн.): Silhouette={sil_real:.4f}, ARI={ari_real:.4f}, AMI={ami_real:.4f}")


# ===========================================================
# 3. ВИЗУАЛИЗАЦИЯ
# ===========================================================

# --- График 1: Предсказания vs Истина (реальный датасет) ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, y_pred, title in zip(
    axes,
    [y_pred_etr_real, y_pred_lr_real],
    ['ExtraTreesRegressor (реальный)', 'LinearRegression (реальный)']
):
    ax.scatter(y_val, y_pred, alpha=0.3, s=8)
    lims = [min(y_val.min(), y_pred.min()), max(y_val.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5)
    ax.set_xlabel('Истинная цена, руб.')
    ax.set_ylabel('Предсказанная цена, руб.')
    ax.set_title(title)
plt.tight_layout()
plt.savefig('plot_predictions.png', dpi=120)
plt.close()
print("\nСохранено: plot_predictions.png")

# --- График 2: Остатки ---
res_etr = y_val - y_pred_etr_real
res_lr  = y_val - y_pred_lr_real
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, res, title in zip(
    axes,
    [res_etr, res_lr],
    ['Остатки ExtraTrees', 'Остатки LinearRegression']
):
    ax.hist(res, bins=50, edgecolor='black', color='steelblue')
    ax.set_xlabel('Остатки, руб.')
    ax.set_ylabel('Частота')
    ax.set_title(title)
plt.tight_layout()
plt.savefig('plot_residuals.png', dpi=120)
plt.close()
print("Сохранено: plot_residuals.png")

# --- График 3: PCA + кластеры ---
pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_all)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_real, cmap='viridis', alpha=0.5, s=5)
axes[0].set_title('KMeans — кластеры')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_bin_real, cmap='coolwarm', alpha=0.5, s=5)
axes[1].set_title('Истинные метки (выше/ниже медианы цены)')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
plt.tight_layout()
plt.savefig('plot_clusters.png', dpi=120)
plt.close()
print("Сохранено: plot_clusters.png")

# --- График 4: Важность признаков ExtraTrees (реальный датасет) ---
etr_model = etr_pipe.named_steps['regressor']
feature_names = (
    numeric_features
    + list(etr_pipe.named_steps['preprocess']
           .named_transformers_['cat']
           .get_feature_names_out(categorical_features))
)
importances = etr_model.feature_importances_
top_n = 10
idx = np.argsort(importances)[::-1][:top_n]
plt.figure(figsize=(10, 4))
plt.bar(range(top_n), importances[idx], color='steelblue')
plt.xticks(range(top_n), [feature_names[i] for i in idx], rotation=30, ha='right')
plt.ylabel('Важность')
plt.title('Топ-10 важных признаков — ExtraTreesRegressor')
plt.tight_layout()
plt.savefig('plot_feature_importance.png', dpi=120)
plt.close()
print("Сохранено: plot_feature_importance.png")

# ===========================================================
# 4. СВОДНАЯ ТАБЛИЦА
# ===========================================================
print("\n" + "=" * 70)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 70)

summary = pd.DataFrame({
    'Датасет': ['Синтетический', 'Синтетический', 'Реальный', 'Реальный'],
    'Модель':  ['ExtraTreesRegressor', 'LinearRegression',
                'ExtraTreesRegressor', 'LinearRegression'],
    'R²': [
        r2_score(y_syn_val, y_pred_etr_syn),
        r2_score(y_syn_val, y_pred_lr_syn),
        r2_score(y_val, y_pred_etr_real),
        r2_score(y_val, y_pred_lr_real),
    ],
    'RMSE': [
        np.sqrt(mean_squared_error(y_syn_val, y_pred_etr_syn)),
        np.sqrt(mean_squared_error(y_syn_val, y_pred_lr_syn)),
        np.sqrt(mean_squared_error(y_val, y_pred_etr_real)),
        np.sqrt(mean_squared_error(y_val, y_pred_lr_real)),
    ],
    'MAE': [
        mean_absolute_error(y_syn_val, y_pred_etr_syn),
        mean_absolute_error(y_syn_val, y_pred_lr_syn),
        mean_absolute_error(y_val, y_pred_etr_real),
        mean_absolute_error(y_val, y_pred_lr_real),
    ],
    'Время (с)': [time_etr_syn, time_lr_syn, time_etr_real, time_lr_real],
})
print(summary.to_string(index=False))

print("\nКластеризация (KMeans, k=2):")
clust_summary = pd.DataFrame({
    'Датасет':   ['Синтетический', 'Реальный'],
    'Silhouette': [sil_syn, sil_real],
    'ARI':        [ari_syn, ari_real],
    'AMI':        [ami_syn, ami_real],
})
print(clust_summary.to_string(index=False))