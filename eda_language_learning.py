"""
EDA - Language Learning Fluency (Kaggle)
========================================

Análisis exploratorio completo de:
  - data/language_learning_fluency.csv  (45.000 aprendices x 18 columnas)
  - data/data_dictionary_language.csv   (diccionario de datos)

Uso:
    pip install -r requirements.txt
    python eda_language_learning.py

Salidas:
    outputs/figures/*.png   -> gráficos
    outputs/tables/*.csv    -> tablas de resultados
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # permite ejecutar sin interfaz gráfica
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "language_learning_fluency.csv"
DICT_PATH = BASE_DIR / "data" / "data_dictionary_language.csv"
FIG_DIR = BASE_DIR / "outputs" / "figures"
TAB_DIR = BASE_DIR / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CEFR_ORDER = ["A0", "A1", "A2", "B1", "B2", "C1", "C2"]
TARGETS = ["reached_fluency", "dropped_out"]
ID_COL = "learner_id"

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title.upper())
    print("=" * 80)


def save_fig(name: str) -> None:
    plt.savefig(FIG_DIR / f"{name}.png")
    plt.close()
    print(f"  -> figura guardada: outputs/figures/{name}.png")


# ---------------------------------------------------------------------------
# 1. Carga de datos y diccionario
# ---------------------------------------------------------------------------
section("1. Carga de datos")
df = pd.read_csv(DATA_PATH)
dictionary = pd.read_csv(DICT_PATH)

print(f"Dataset: {df.shape[0]:,} filas x {df.shape[1]} columnas")
print(f"Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print("\nPrimeras filas:")
print(df.head())

print("\nDiccionario de datos:")
print(dictionary.to_string(index=False))

# Validar que diccionario y dataset describen las mismas columnas
missing_in_dict = set(df.columns) - set(dictionary["column"])
missing_in_data = set(dictionary["column"]) - set(df.columns)
print(f"\nColumnas sin documentar: {missing_in_dict or 'ninguna'}")
print(f"Columnas documentadas pero ausentes: {missing_in_data or 'ninguna'}")

# Comparar tipos declarados vs. tipos reales
type_checks = {
    "int": pd.api.types.is_integer_dtype,
    "float": pd.api.types.is_float_dtype,
    "string": lambda s: pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s),
}
types = dictionary.set_index("column")[["type"]].rename(columns={"type": "tipo_declarado"})
types["tipo_real"] = df.dtypes.astype(str)
types["coincide"] = [type_checks[t](df[c]) for c, t in types["tipo_declarado"].items()]
print("\nTipos declarados vs. reales:")
print(types)

# ---------------------------------------------------------------------------
# 2. Clasificación de variables
# ---------------------------------------------------------------------------
section("2. Tipos de variables")
binary_cols = [c for c in df.columns if df[c].nunique() == 2 and c not in TARGETS]
categorical_cols = ["cefr_level", "fsi_category"]
numeric_cols = [
    c
    for c in df.select_dtypes(include=np.number).columns
    if c not in TARGETS + binary_cols + categorical_cols
]
print(f"Identificador : {ID_COL}")
print(f"Objetivos     : {TARGETS}")
print(f"Binarias      : {binary_cols}")
print(f"Categóricas   : {categorical_cols}")
print(f"Numéricas     : {numeric_cols}")

# ---------------------------------------------------------------------------
# 3. Calidad de datos: faltantes, duplicados, rangos
# ---------------------------------------------------------------------------
section("3. Valores faltantes y calidad")
missing = pd.DataFrame(
    {
        "faltantes": df.isna().sum(),
        "pct_faltantes": (df.isna().mean() * 100).round(2),
        "vacios_str": (df.astype(str).apply(lambda s: s.str.strip() == "")).sum(),
        "valores_unicos": df.nunique(),
    }
)
print(missing)
missing.to_csv(TAB_DIR / "valores_faltantes.csv")
print(f"\nTotal de celdas faltantes: {int(df.isna().sum().sum())}")
print(f"Filas duplicadas: {df.duplicated().sum()}")
print(f"IDs duplicados: {df[ID_COL].duplicated().sum()}")

plt.figure(figsize=(10, 4))
sns.heatmap(df.isna().T, cbar=False, cmap="viridis", yticklabels=True, xticklabels=False)
plt.title("Missing values map (yellow = missing)")
save_fig("01_mapa_faltantes")

# Rangos esperados según el diccionario de datos
expected_ranges = {
    "fsi_category": (1, 4),
    "active_use_share": (0, 1),
    "speaking_practice": (0, 10),
    "feedback_quality": (0, 10),
    "motivation": (0, 10),
    "consistency": (0, 10),
    "accent_nativelike": (0, 10),
}
print("\nValores fuera de rango esperado:")
for col, (lo, hi) in expected_ranges.items():
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"  {col:<22} [{lo}, {hi}] -> {n_out} fuera de rango")

non_negative = ["total_study_hours", "immersion_months", "comprehensible_input_hours",
                "prior_languages", "age_started"]
print("Valores negativos en variables no negativas:",
      {c: int((df[c] < 0).sum()) for c in non_negative})

# Consistencia lógica entre columnas
fluent_from_cefr = df["cefr_level"].isin(["B2", "C1", "C2"]).astype(int)
print(f"\nreached_fluency coherente con cefr_level (>=B2): "
      f"{(fluent_from_cefr == df['reached_fluency']).mean():.2%}")
both = ((df["reached_fluency"] == 1) & (df["dropped_out"] == 1)).sum()
print(f"Aprendices con reached_fluency=1 y dropped_out=1 a la vez: {both}")
print("dropped_out por nivel CEFR:")
print(df.groupby("cefr_level")["dropped_out"].mean().reindex(CEFR_ORDER).round(3))

# ---------------------------------------------------------------------------
# 4. Estadísticas descriptivas
# ---------------------------------------------------------------------------
section("4. Estadísticas descriptivas")
desc = df[numeric_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
desc["skew"] = df[numeric_cols].skew()
desc["kurtosis"] = df[numeric_cols].kurtosis()
desc["cv"] = desc["std"] / desc["mean"]
desc["pct_ceros"] = (df[numeric_cols] == 0).mean() * 100
print(desc.round(3))
desc.round(4).to_csv(TAB_DIR / "estadisticas_descriptivas.csv")

# Outliers por criterio IQR
q1, q3 = df[numeric_cols].quantile(0.25), df[numeric_cols].quantile(0.75)
iqr = q3 - q1
outliers = ((df[numeric_cols] < q1 - 1.5 * iqr) | (df[numeric_cols] > q3 + 1.5 * iqr)).sum()
outliers = pd.DataFrame({"n_outliers": outliers, "pct": (outliers / len(df) * 100).round(2)})
print("\nOutliers (regla 1.5*IQR):")
print(outliers.sort_values("n_outliers", ascending=False))
outliers.to_csv(TAB_DIR / "outliers_iqr.csv")

print("\nVariables categóricas / binarias:")
for col in categorical_cols + binary_cols + TARGETS:
    vc = df[col].value_counts(normalize=True).sort_index()
    if col == "cefr_level":
        vc = vc.reindex(CEFR_ORDER)
    print(f"\n{col}:\n{(vc * 100).round(2).to_string()}")

# ---------------------------------------------------------------------------
# 5. Análisis univariado
# ---------------------------------------------------------------------------
section("5. Análisis univariado")
n = len(numeric_cols)
ncols = 3
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.6 * nrows))
for ax, col in zip(axes.flat, numeric_cols):
    sns.histplot(df[col], bins=40, kde=True, ax=ax)
    ax.set_title(f"{col}\nskew={df[col].skew():.2f}")
    ax.set_xlabel("")
for ax in axes.flat[n:]:
    ax.set_visible(False)
fig.suptitle("Distribution of numeric variables", y=1.01, fontsize=14)
plt.tight_layout()
save_fig("02_histogramas_numericas")

fig, axes = plt.subplots(nrows, ncols, figsize=(15, 2.6 * nrows))
for ax, col in zip(axes.flat, numeric_cols):
    sns.boxplot(x=df[col], ax=ax, color="lightsteelblue", fliersize=1)
    ax.set_title(col)
    ax.set_xlabel("")
for ax in axes.flat[n:]:
    ax.set_visible(False)
fig.suptitle("Boxplots (outlier detection)", y=1.01, fontsize=14)
plt.tight_layout()
save_fig("03_boxplots_numericas")

cat_plot_cols = categorical_cols + binary_cols + TARGETS
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flat, cat_plot_cols):
    order = CEFR_ORDER if col == "cefr_level" else sorted(df[col].unique())
    sns.countplot(data=df, x=col, order=order, ax=ax)
    total = len(df)
    for p in ax.patches:
        ax.annotate(f"{p.get_height() / total:.1%}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_title(col)
for ax in axes.flat[len(cat_plot_cols):]:
    ax.set_visible(False)
plt.tight_layout()
save_fig("04_variables_categoricas")

# ---------------------------------------------------------------------------
# 6. Análisis de las variables objetivo
# ---------------------------------------------------------------------------
section("6. Variables objetivo")
ct = pd.crosstab(df["reached_fluency"], df["dropped_out"], margins=True)
print("Tabla cruzada reached_fluency x dropped_out:")
print(ct)

# Resultado combinado en 3 clases (más interpretable que dos binarias)
df["outcome"] = np.select(
    [df["reached_fluency"] == 1, df["dropped_out"] == 1],
    ["fluent", "dropped"],
    default="in_progress",
)
print("\nResultado combinado:")
print(df["outcome"].value_counts(normalize=True).round(3))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
cefr_outcome = pd.crosstab(df["cefr_level"], df["outcome"], normalize="index").reindex(CEFR_ORDER)
cefr_outcome[["fluent", "in_progress", "dropped"]].plot(
    kind="bar", stacked=True, ax=axes[0], color=["#2a9d8f", "#e9c46a", "#e76f51"])
axes[0].set_title("Outcome by CEFR level")
axes[0].set_ylabel("Share")
axes[0].legend(title="outcome", bbox_to_anchor=(1, 1))
fsi_rates = df.groupby("fsi_category")[TARGETS].mean()
fsi_rates.plot(kind="bar", ax=axes[1], color=["#2a9d8f", "#e76f51"])
axes[1].set_title("Fluency and dropout rate by FSI difficulty")
axes[1].set_ylabel("Rate")
plt.tight_layout()
save_fig("05_objetivos")

# ---------------------------------------------------------------------------
# 7. Correlaciones
# ---------------------------------------------------------------------------
section("7. Correlaciones")
df["cefr_ordinal"] = df["cefr_level"].map({lvl: i for i, lvl in enumerate(CEFR_ORDER)})
corr_cols = numeric_cols + categorical_cols[1:] + binary_cols + ["cefr_ordinal"] + TARGETS
corr = df[corr_cols].corr(method="spearman")

plt.figure(figsize=(13, 11))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, annot_kws={"size": 7}, linewidths=0.4)
plt.title("Correlation matrix (Spearman)")
save_fig("06_matriz_correlacion")

target_corr = corr[TARGETS + ["cefr_ordinal"]].drop(TARGETS + ["cefr_ordinal"])
target_corr = target_corr.reindex(target_corr["reached_fluency"].abs().sort_values(ascending=False).index)
print("Correlación de Spearman con los objetivos:")
print(target_corr.round(3))
target_corr.round(4).to_csv(TAB_DIR / "correlacion_con_objetivos.csv")

# Pares de predictores muy correlacionados (posible multicolinealidad)
features_only = corr.drop(index=TARGETS + ["cefr_ordinal"], columns=TARGETS + ["cefr_ordinal"])
pairs = (features_only.where(np.triu(np.ones(features_only.shape, dtype=bool), k=1))
         .stack().rename("rho").reset_index())
pairs = pairs[pairs["rho"].abs() > 0.5].sort_values("rho", key=abs, ascending=False)
print("\nPares de predictores con |rho| > 0.5:")
print(pairs.to_string(index=False) if len(pairs) else "  ninguno")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, target in zip(axes, TARGETS):
    s = target_corr[target]
    colors = ["#2a9d8f" if v > 0 else "#e76f51" for v in s]
    ax.barh(s.index, s.values, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(f"Correlation with {target}")
axes[0].invert_yaxis()
plt.tight_layout()
save_fig("07_correlacion_objetivos")

# ---------------------------------------------------------------------------
# 8. Análisis bivariado frente a los objetivos
# ---------------------------------------------------------------------------
section("8. Análisis bivariado")
key_numeric = ["total_study_hours", "comprehensible_input_hours", "active_use_share",
               "immersion_months", "motivation", "consistency"]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.flat, key_numeric):
    sns.boxplot(data=df, x="outcome", y=col, order=["dropped", "in_progress", "fluent"],
                hue="outcome", palette={"fluent": "#2a9d8f", "in_progress": "#e9c46a",
                                        "dropped": "#e76f51"},
                legend=False, ax=ax, fliersize=1)
    ax.set_title(col)
    ax.set_xlabel("")
plt.tight_layout()
save_fig("08_boxplots_por_resultado")

print("Medias por resultado:")
means_by_outcome = df.groupby("outcome")[numeric_cols].mean().T.round(2)
print(means_by_outcome)
means_by_outcome.to_csv(TAB_DIR / "medias_por_resultado.csv")

# Tasa de fluidez por deciles de horas de estudio y dificultad del idioma
df["horas_bin"] = pd.qcut(df["total_study_hours"], q=10, duplicates="drop")
pivot = df.pivot_table(index="horas_bin", columns="fsi_category",
                       values="reached_fluency", aggfunc="mean", observed=True)
plt.figure(figsize=(10, 5))
for fsi in pivot.columns:
    plt.plot(range(len(pivot)), pivot[fsi], marker="o", label=f"FSI {fsi}")
plt.xticks(range(len(pivot)), [str(i) for i in pivot.index], rotation=45, ha="right")
plt.ylabel("Fluency rate (>= B2)")
plt.xlabel("total_study_hours deciles")
plt.title("Fluency vs study hours by language difficulty")
plt.legend()
save_fig("09_fluidez_horas_fsi")
print("\nTasa de fluidez por decil de horas x FSI:")
print(pivot.round(2))

# Efecto de la calidad del estudio: active_use_share en cuartiles
df["active_q"] = pd.qcut(df["active_use_share"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
pivot_active = df.pivot_table(index="horas_bin", columns="active_q",
                              values="reached_fluency", aggfunc="mean", observed=True)
plt.figure(figsize=(10, 5))
for q in pivot_active.columns:
    plt.plot(range(len(pivot_active)), pivot_active[q], marker="o", label=str(q))
plt.xticks(range(len(pivot_active)), [str(i) for i in pivot_active.index],
           rotation=45, ha="right")
plt.legend(title="active_use_share")
plt.title("Fluency vs study hours by active-use quartile")
plt.ylabel("Fluency rate")
plt.xlabel("total_study_hours deciles")
save_fig("10_fluidez_horas_uso_activo")

# Abandono vs motivación y consistencia
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, col in zip(axes, ["motivation", "consistency"]):
    bins = pd.cut(df[col], bins=10)
    rate = df.groupby(bins, observed=True)["dropped_out"].mean()
    ax.plot(range(len(rate)), rate.values, marker="o", color="#e76f51")
    ax.set_xticks(range(len(rate)))
    ax.set_xticklabels([f"{i.mid:.1f}" for i in rate.index])
    ax.set_title(f"Dropout rate vs {col}")
    ax.set_ylabel("dropped_out")
plt.tight_layout()
save_fig("11_abandono_motivacion_consistencia")

# Edad de inicio: afecta al acento, no a la fluidez (según el diccionario)
df["edad_bin"] = pd.cut(df["age_started"], bins=[0, 12, 18, 25, 35, 45, 70])
age_eff = df.groupby("edad_bin", observed=True)[["accent_nativelike", "reached_fluency"]].mean()
print("\nEfecto de la edad de inicio:")
print(age_eff.round(3))
fig, ax1 = plt.subplots(figsize=(9, 4.5))
ax1.plot(age_eff.index.astype(str), age_eff["accent_nativelike"], marker="o", color="#264653",
         label="accent_nativelike")
ax1.set_ylabel("accent_nativelike (0-10)")
ax2 = ax1.twinx()
ax2.plot(age_eff.index.astype(str), age_eff["reached_fluency"], marker="s", color="#2a9d8f",
         label="reached_fluency")
ax2.set_ylabel("Fluency rate")
ax2.set_ylim(0, 1)
ax2.grid(False)
fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88))
plt.title("Starting age: accent vs fluency")
save_fig("12_edad_acento_fluidez")

# ---------------------------------------------------------------------------
# 9. Importancia de variables
# ---------------------------------------------------------------------------
section("9. Importancia de variables")
# Se excluyen learner_id (identificador) y cefr_level (define reached_fluency => fuga de datos).
# accent_nativelike es un resultado del aprendizaje, no un factor previo; también se excluye.
feature_cols = ["fsi_category"] + binary_cols + numeric_cols
feature_cols = [c for c in feature_cols if c != "accent_nativelike"]
X = df[feature_cols]

importance = {}
for target in TARGETS:
    y = df[target]
    mi = mutual_info_classif(X, y, discrete_features=[c in binary_cols or c == "fsi_category"
                                                      for c in feature_cols],
                             random_state=RANDOM_STATE)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y,
                                              random_state=RANDOM_STATE)
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=20,
                                n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_tr, y_tr)
    acc = rf.score(X_te, y_te)
    perm = permutation_importance(rf, X_te, y_te, n_repeats=5, scoring="roc_auc",
                                  random_state=RANDOM_STATE, n_jobs=-1)
    print(f"\n[{target}] accuracy RandomForest (test): {acc:.3f} "
          f"(baseline clase mayoritaria: {1 - y_te.mean():.3f})")

    imp = pd.DataFrame({
        "mutual_info": mi,
        "rf_gini": rf.feature_importances_,
        "permutation_auc": perm.importances_mean,
    }, index=feature_cols)
    # Ranking promedio entre los tres métodos
    imp["rank_medio"] = imp.rank(ascending=False).mean(axis=1)
    imp = imp.sort_values("rank_medio")
    importance[target] = imp
    print(imp.round(4))
    imp.round(5).to_csv(TAB_DIR / f"importancia_{target}.csv")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, target in zip(axes, TARGETS):
    imp = importance[target].sort_values("permutation_auc")
    ax.barh(imp.index, imp["permutation_auc"], color="#264653")
    ax.set_title(f"Permutation importance (AUC drop)\ntarget: {target}")
    ax.set_xlabel("Mean decrease in ROC-AUC")
plt.tight_layout()
save_fig("13_importancia_variables")

# ---------------------------------------------------------------------------
# 10. Resumen
# ---------------------------------------------------------------------------
section("10. Resumen")
top_fluency = importance["reached_fluency"].index[:5].tolist()
top_dropout = importance["dropped_out"].index[:5].tolist()
print(f"- {df.shape[0]:,} registros, {df.shape[1] - 5} columnas originales, "
      f"{int(df[feature_cols].isna().sum().sum())} valores faltantes, 0 duplicados.")
print(f"- Fluidez alcanzada: {df['reached_fluency'].mean():.1%} | "
      f"Abandono: {df['dropped_out'].mean():.1%} | "
      f"En progreso: {(df['outcome'] == 'in_progress').mean():.1%}")
print(f"- Top 5 variables para reached_fluency: {top_fluency}")
print(f"- Top 5 variables para dropped_out:     {top_dropout}")
print("- cefr_level determina reached_fluency (>=B2): no usar como predictor.")
print(f"\nFiguras en: {FIG_DIR}\nTablas en:  {TAB_DIR}")
