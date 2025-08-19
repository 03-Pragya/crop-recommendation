# =========================================================================
# Back-end: Full Pipeline for Data Analysis and Model Training
# =========================================================================

# -------- Imports --------
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os 

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# Create a folder to save the plots if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# -------- Helper to show plots with title, gap, and a printed inference --------
def show_plot(title, plot_fn, inference, filename=None, figsize=(10, 6)):
    print("\n" + "="*100)
    print(f"Figure: {title}")
    print("="*100)
    plt.figure(figsize=figsize)
    plot_fn()
    plt.tight_layout()
    # Save the plot to the specified filename
    if filename:
        plt.savefig(os.path.join('plots', filename))
    plt.show()
    print(f"Inference: {inference}")
    print("\n")

# -------- Helper: classification report as a tidy, aligned table --------
def make_report_table(y_true, y_pred, class_names):
    rpt = classification_report(
        y_true, y_pred, target_names=list(class_names), output_dict=True, zero_division=0
    )
    rows = []
    for k, v in rpt.items():
        if k == "accuracy":
            continue
        rows.append((k, v.get("precision", np.nan), v.get("recall", np.nan),
                     v.get("f1-score", np.nan), int(v.get("support", 0))))
    rep_df = pd.DataFrame(rows, columns=["class", "precision", "recall", "f1-score", "support"]).set_index("class")
    # Add accuracy row
    rep_df.loc["accuracy", ["precision","recall"]] = np.nan
    rep_df.loc["accuracy", "f1-score"] = rpt["accuracy"]
    rep_df.loc["accuracy", "support"]  = len(y_true)
    # Order: classes, macro avg, weighted avg, accuracy
    ordered_index = list(class_names) + ["macro avg", "weighted avg", "accuracy"]
    rep_df = rep_df.reindex([idx for idx in ordered_index if idx in rep_df.index])
    # Formatting
    return rep_df

# -------- Helper: plot confusion matrix with numbers and return a short inference --------
def plot_confusion_with_inference(y_true, y_pred, class_names, title):
    cm = confusion_matrix(y_true, y_pred)
    def _plot():
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=class_names, yticklabels=class_names, annot_kws={"size":8}
        )
        plt.title(title, fontsize=16)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
    # Find the top confusion off-diagonal for inference
    cm_off = cm.copy().astype(float)
    np.fill_diagonal(cm_off, 0)
    if cm_off.sum() > 0:
        i, j = divmod(np.argmax(cm_off), cm_off.shape[1])
        conf_pair = f"Most confusion: actual '{class_names[i]}' predicted as '{class_names[j]}' ({int(cm_off[i, j])} cases)."
    else:
        conf_pair = "No off-diagonal confusions observed."
    return cm, _plot, conf_pair

# ===========================================
# 1) Load Dataset and Initial Inspection
# ===========================================
df = pd.read_csv("/content/drive/MyDrive/Project_Crop/Crop_recommendation.csv")

print("\n" + "="*100)
print("DATASET INFORMATION")
print("="*100)
print(df.info())

print("\n" + "="*100)
print("STATISTICAL SUMMARY")
print("="*100)
print(df.describe())

print("\n" + "="*100)
print("FIRST 5 ROWS")
print("="*100)
print(df.head())

# ===========================================
# 2) EDA with Titled Figures and Inferences
# ===========================================
show_plot(
    "Missing Values Heatmap",
    lambda: sns.heatmap(df.isnull(), cbar=False, cmap="coolwarm"),
    "No missing values visible; dataset appears complete.",
    filename="Missing_Values_Heatmap.png",
    figsize=(10, 2)
)

show_plot(
    "Temperature Distribution",
    lambda: sns.histplot(df['temperature'], bins=15, kde=True),
    "Temperature clusters roughly around mid-20s to low-30s °C, indicating moderate ambient conditions.",
    filename="Temperature_Distribution.png"
)

show_plot(
    "pH Distribution",
    lambda: sns.histplot(df['ph'], bins=15, kde=True),
    "Soil pH centers near neutral (around 6.5–7.0), suitable for many crops.",
    filename="pH_Distribution.png"
)

show_plot(
    "Crop Distribution (Class Balance)",
    lambda: sns.countplot(y='label', data=df, order=sorted(df['label'].unique())),
    "Classes appear fairly balanced across different crops in this dataset.",
    filename="Crop_Distribution.png"
)

feature_cols = ['N','P','K','temperature','humidity','ph','rainfall']
show_plot(
    "Feature Correlation Heatmap",
    lambda: sns.heatmap(df[feature_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f"),
    "Humidity and rainfall show mild relationships; nutrients (N, P, K) are comparatively weakly correlated.",
    filename="Feature_Correlation_Heatmap.png",
    figsize=(9, 7)
)

# ===========================================
# 3) Preprocessing
# ===========================================
c = df.label.astype('category')
class_names = list(c.cat.categories)
targets = dict(enumerate(class_names))
df['target'] = c.cat.codes

X = df[feature_cols]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42, test_size=0.2
)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ===========================================
# 4) Model Definitions + Hyperparameter Grids
#    (All five models handled identically)
# ===========================================
models = {
    "KNN": (
        KNeighborsClassifier(),
        {"n_neighbors": [3, 5, 7, 9, 11]},
        True  # needs scaling
    ),
    "SVM (RBF)": (
        SVC(probability=True),
        {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
        True  # needs scaling
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {"max_depth": [None, 5, 10, 20]},
        False
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {"n_estimators": [100, 200], "max_depth": [None, 5, 10]},
        False
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
        False
    )
}

# ===========================================
# 5) Train, Tune, and Evaluate Each Classifier
#    (Print aligned tables; plot confusion with numbers; add inferences)
# ===========================================
results          = {}
best_estimators  = {}

for name, (model, params, needs_scaling) in models.items():
    print("\n" + "="*100)
    print(f"{name} — Training, Hyperparameter Tuning, and Evaluation")
    print("="*100)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(model, params, cv=cv, scoring="f1_macro", n_jobs=-1)

    if needs_scaling:
        grid.fit(X_train_scaled, y_train)
        y_pred = grid.predict(X_test_scaled)
    else:
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    rpt_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    f1_macro = rpt_dict["macro avg"]["f1-score"]

    results[name] = {"Accuracy": acc, "F1-macro": f1_macro}
    best_estimators[name] = grid.best_estimator_

    # Summary info (as in your reference image)
    print(f"Best Parameters  : {grid.best_params_}")
    print(f"CV Best F1-macro : {grid.best_score_:.4f}")
    print(f"Test Accuracy    : {acc:.4f}")
    print(f"Test F1-macro    : {f1_macro:.4f}")

    # Report table (aligned)
    report_df = make_report_table(y_test, y_pred, class_names)
    with pd.option_context("display.float_format", "{:,.4f}".format):
        print("\nClassification Report (precision | recall | f1-score | support)")
        print(report_df.to_string(justify="center"))

    # Confusion matrix with numbers + inference
    cm, cm_plot_fn, cm_infer = plot_confusion_with_inference(
        y_test, y_pred, class_names, title=f"Confusion Matrix — {name}"
    )
    show_plot(
        f"Confusion Matrix — {name}",
        cm_plot_fn,
        cm_infer,
        filename=f"{name}_Confusion_Matrix.png",
        figsize=(14, 12)
    )

# ===========================================
# 6) Model Comparison: Table and Bar Chart + Improvement Numbers
# ===========================================
results_df = pd.DataFrame(results).T
results_numeric = results_df.copy()  # numeric copy for selection/math

# Pretty/aligned table for display
pretty_df = results_df.copy()
for col in pretty_df.columns:
    pretty_df[col] = pretty_df[col].map(lambda x: f"{x:,.4f}")

print("\n" + "="*100)
print("MODEL COMPARISON (Accuracy and F1-macro)")
print("="*100)
print(pretty_df.to_string(justify="center"))

# Select best by F1-macro
best_model_name = results_numeric["F1-macro"].idxmax()
best_model      = best_estimators[best_model_name]

# Compute improvement over second best (by F1-macro)
sorted_f1 = results_numeric.sort_values("F1-macro", ascending=False)
runner_up_name = sorted_f1.index[1]
best_f1     = sorted_f1.iloc[0]["F1-macro"]
runner_f1   = sorted_f1.iloc[1]["F1-macro"]
f1_impr_abs = best_f1 - runner_f1
f1_impr_pct = (f1_impr_abs / runner_f1) * 100 if runner_f1 > 0 else np.inf

best_acc     = results_numeric.loc[best_model_name, "Accuracy"]
runner_acc   = results_numeric.loc[runner_up_name, "Accuracy"]
acc_impr_abs = best_acc - runner_acc
acc_impr_pct = (acc_impr_abs / runner_acc) * 100 if runner_acc > 0 else np.inf

def _barplot_models():
    sns.barplot(x=results_numeric.index, y=results_numeric["F1-macro"])
    plt.title("Model F1-macro Comparison")
    plt.xlabel("Model")
    plt.ylabel("F1-macro")
    plt.xticks(rotation=20)
    # Highlight best model line
    plt.axhline(best_f1, linestyle="--", linewidth=1)

show_plot(
    "Model F1-macro Comparison",
    _barplot_models,
    f"Best model: {best_model_name}. F1-macro improvement over {runner_up_name}: "
    f"{f1_impr_abs:.4f} ({f1_impr_pct:.2f}%). Accuracy improvement: "
    f"{acc_impr_abs:.4f} ({acc_impr_pct:.2f}%).",
    filename="Model_Comparison_Barplot.png",
    figsize=(9, 6)
)

print("\n" + "="*100)
print(f"BEST MODEL SELECTED: {best_model_name}")
print("="*100)
print(f"F1-macro improvement over {runner_up_name}: {f1_impr_abs:.4f} ({f1_impr_pct:.2f}%)")
print(f"Accuracy improvement over {runner_up_name}: {acc_impr_abs:.4f} ({acc_impr_pct:.2f}%)")

# Feature importance if tree-based
if best_model_name in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
    def _feat_imp():
        imp = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values()
        imp.plot(kind="barh")
        plt.title(f"Feature Importance — {best_model_name}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
    # Simple inference: top feature
    top_feat = feature_cols[np.argmax(best_model.feature_importances_)]
    show_plot(
        f"Feature Importance — {best_model_name}",
        _feat_imp,
        f"Most influential feature: {top_feat}. Consider focusing on this variable when assessing crop suitability.",
        filename=f"{best_model_name}_Feature_Importance.png"
    )

# =========================================================================
# SAVE THE BEST MODEL, THE SCALER, AND THE CLASS NAMES
# =========================================================================
print("\n" + "="*100)
print("SAVING MODEL AND SCALER")
print("="*100)

joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save the mapping from numerical codes to crop names
targets_df = pd.DataFrame(list(targets.items()), columns=['code', 'crop']).set_index('code')
targets_df.to_csv("targets.csv")

print("Best model, scaler, and target names saved successfully!")
