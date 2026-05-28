import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import time

# 1. ИСХОДНЫЙ ДАТАСЕТ

data = pd.DataFrame({
    'продукт':  ['Яблоко',  'Салат', 'Бекон',    'Банан',   'Орехи',
                 'Рыба',    'Сыр',   'Виноград',  'Морковь', 'Апельсин'],
    'сладость': [7,          2,       1,           9,         1,
                 1,          1,       8,           2,         6],
    'хруст':    [7,          5,       2,           1,         5,
                 1,          1,       1,           8,         1],
    'класс':    ['Фрукт',   'Овощ',  'Протеин',  'Фрукт',   'Протеин',
                 'Протеин', 'Протеин','Фрукт',    'Овощ',    'Фрукт']
})

print("=== ИСХОДНЫЙ ДАТАСЕТ ===")
print(data.to_string(index=False))

X = data[['сладость', 'хруст']].values
y = data['класс'].values

# 2. ТЕСТОВЫЕ ОБЪЕКТЫ

test = pd.DataFrame({
    'продукт':  ['Томат',  'Арбуз', 'Сельдерей'],
    'сладость': [4,         9,       2],
    'хруст':    [5,         4,       9],
})

print("\n=== ТЕСТОВЫЕ ОБЪЕКТЫ ===")
print(test.to_string(index=False))

X_test = test[['сладость', 'хруст']].values
test_names = test['продукт'].tolist()

# 3. СОБСТВЕННЫЙ k-NN
class CustomKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        results = []
        for x in X:
            dists = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_idx = np.argsort(dists)[:self.k]
            k_labels = self.y_train[k_idx]
            labels, counts = np.unique(k_labels, return_counts=True)
            max_count = counts.max()
            candidates = labels[counts == max_count]
            if len(candidates) == 1:
                results.append(candidates[0])
            else:
                for label in k_labels:
                    if label in candidates:
                        results.append(label)
                        break
        return np.array(results)

# 4. СРАВНЕНИЕ CUSTOM vs SKLEARN
def compare(X_train, y_train, X_test, k=3, test_names=None):
    custom = CustomKNN(k=k).fit(X_train, y_train)
    sklearn_knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean').fit(X_train, y_train)

    cp = custom.predict(X_test)
    sp = sklearn_knn.predict(X_test)

    print(f"{'Продукт':<18} {'Custom k-NN':<15} {'Sklearn k-NN'}")
    print("-" * 50)
    names = test_names if test_names else [str(i) for i in range(len(X_test))]
    for i, nm in enumerate(names):
        mark = "" if cp[i] == sp[i] else "  ← расхождение"
        print(f"{nm:<18} {cp[i]:<15} {sp[i]}{mark}")

    print(f"\nПолное совпадение: {np.all(cp == sp)}")
    return cp, sp

print("\n=== ЭКСПЕРИМЕНТ 1: 3 класса, k=3 ===")
cp1, sp1 = compare(X, y, X_test, k=3, test_names=test_names)

# 5. ВИЗУАЛИЗАЦИЯ РАЗДЕЛЯЮЩИХ ПОВЕРХНОСТЕЙ
def plot_boundaries(model, X, y, test_pts=None, test_names=None, title="", ax=None):
    if ax is None:
        ax = plt.gca()
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    le = LabelEncoder()
    Z_enc = le.fit_transform(Z).reshape(xx.shape)
    classes = np.unique(y)
    palette = sns.color_palette("Set2", len(classes))
    ax.contourf(xx, yy, Z_enc, alpha=0.3,
                levels=np.arange(len(classes)+1)-0.5,
                colors=palette)
    for cls in classes:
        mask = y == cls
        ax.scatter(X[mask,0], X[mask,1], label=cls, s=90,
                   edgecolor='black', lw=1, color=palette[list(classes).index(cls)])
    if test_pts is not None:
        ax.scatter(test_pts[:,0], test_pts[:,1], marker='*', s=230,
                   c='red', edgecolor='black', lw=1, label='Тест', zorder=5)
        for i, (px, py) in enumerate(test_pts):
            ax.annotate(test_names[i] if test_names else "", (px, py),
                        xytext=(5,5), textcoords='offset points', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
    ax.set_xlabel('Сладость')
    ax.set_ylabel('Хруст')
    ax.set_title(title, fontsize=13)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)

model3 = KNeighborsClassifier(n_neighbors=3, metric='euclidean').fit(X, y)

fig, ax = plt.subplots(figsize=(9, 6))
plot_boundaries(model3, X, y, X_test, test_names,
                "Разделяющие поверхности (3 класса, k=3)", ax)
plt.tight_layout()
plt.show()

# 6. ЭКСПЕРИМЕНТ 2: ДОБАВЛЯЕМ КЛАСС «ЯГОДЫ» (4 класса)
print("\n=== ЭКСПЕРИМЕНТ 2: добавление класса «Ягоды», k=3 ===")

berries = pd.DataFrame({
    'продукт':  ['Клубника', 'Черника', 'Крыжовник'],
    'сладость': [8,           6,         4],
    'хруст':    [6,           4,         7],
    'класс':    ['Ягоды',    'Ягоды',   'Ягоды']
})

data2 = pd.concat([data, berries], ignore_index=True)
X2 = data2[['сладость', 'хруст']].values
y2 = data2['класс'].values

print("\nРасширенный датасет:")
print(data2.to_string(index=False))
print()
compare(X2, y2, X_test, k=3, test_names=test_names)

model4 = KNeighborsClassifier(n_neighbors=3, metric='euclidean').fit(X2, y2)

# График 4 классов
fig, ax = plt.subplots(figsize=(9, 6))
plot_boundaries(model4, X2, y2, X_test, test_names,
                "Разделяющие поверхности (4 класса, k=3)", ax)
plt.tight_layout()
plt.show()

# Сравнение 3 и 4 классов рядом
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
plot_boundaries(model3, X,  y,  X_test, test_names, "3 класса (исходные)", ax1)
plot_boundaries(model4, X2, y2, X_test, test_names, "4 класса (добавлены Ягоды)", ax2)
plt.tight_layout()
plt.show()

# 7. КРОСС-ВАЛИДАЦИЯ LOOCV
print("\n=== КРОСС-ВАЛИДАЦИЯ: Leave-One-Out, 3 класса, k=3 ===")
loo = LeaveOneOut()
k = 3

for label, use_custom in [("Custom k-NN", True), ("Sklearn k-NN", False)]:
    y_true, y_pred = [], []
    t0 = time.perf_counter()
    for tr, te in loo.split(X):
        if use_custom:
            m = CustomKNN(k=k).fit(X[tr], y[tr])
        else:
            m = KNeighborsClassifier(n_neighbors=k, metric='euclidean').fit(X[tr], y[tr])
        p = m.predict(X[te])[0]
        y_true.append(y[te][0])
        y_pred.append(p)
    elapsed = time.perf_counter() - t0
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y))
    print(f"\n{label}: Accuracy={acc:.1%}  F1={f1:.3f}  Время={elapsed:.4f} с")
    print(pd.DataFrame(cm, index=np.unique(y), columns=np.unique(y)))

# ============================================================
# 8. ТЕСТ БЕЛОГО ЯЩИКА — проверка ничьей
# ============================================================
print("\n=== WHITE-BOX ТЕСТ: проверка разрешения ничьей ===")
X_wb = np.array([[2.0, 0.0], [0.0, 2.0], [0.0, -1.9]])
y_wb = np.array(['Фрукт', 'Овощ', 'Протеин'])
pred_wb = CustomKNN(k=3).fit(X_wb, y_wb).predict([[0.0, 0.0]])[0]
print(f"Расстояния от [0,0]: Фрукт=2.00, Овощ=2.00, Протеин=1.90")
print(f"Голоса: 1:1:1 (ничья). Ожидается ближайший = Протеин → {pred_wb}")
print("Тест ПРОЙДЕН ✓" if pred_wb == 'Протеин' else "Тест ПРОВАЛЕН ✗")

# ============================================================
# 9. ТЕСТ ЧЁРНОГО ЯЩИКА — устойчивость к выбросам
# ============================================================
print("\n=== BLACK-BOX ТЕСТ: устойчивость к выбросам ===")
bb_pts = np.array([[0, 0], [10, 10], [-50, -50], [100, 100]])
bb_names = ['(0,0)', '(10,10)', '(-50,-50)', '(100,100)']
try:
    bb_preds = CustomKNN(k=3).fit(X, y).predict(bb_pts)
    for nm, p in zip(bb_names, bb_preds):
        print(f"  Точка {nm:<14} → {p}")
    print("Исключений не возникло — алгоритм устойчив.")
except Exception as e:
    print(f"ОШИБКА: {e}")