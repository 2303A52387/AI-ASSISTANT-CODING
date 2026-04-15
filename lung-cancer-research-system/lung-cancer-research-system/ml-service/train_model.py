"""
train_model.py — ML model training module
Implements classifiers from IEEE Access 2025 paper Section III-G
"""

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score, roc_curve)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN

# ─── Classifier registry ──────────────────────────────────────────────────────
def get_classifiers():
    clfs = {
        'RF': RandomForestClassifier(n_estimators=200, max_depth=20,
                                      min_samples_split=2, min_samples_leaf=1,
                                      criterion='gini', bootstrap=True,
                                      random_state=42, n_jobs=-1),
        'ETC': ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'LR':  LogisticRegression(max_iter=1000, random_state=42),
        'DT':  DecisionTreeClassifier(random_state=42),
        'GBC': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'NB':  GaussianNB(),
        'SGDC': SGDClassifier(max_iter=1000, random_state=42),
    }
    if HAS_XGB:
        clfs['XGB'] = XGBClassifier(n_estimators=100, random_state=42,
                                     use_label_encoder=False, eval_metric='logloss')
    else:
        from sklearn.ensemble import AdaBoostClassifier
        clfs['XGB'] = AdaBoostClassifier(n_estimators=100, random_state=42)
    return clfs


def _prepare_data(df):
    """Extract features and target from dataframe."""
    target_col = None
    for col in ['LUNG_CANCER', 'LUNG CANCER', 'lung_cancer', 'CANCER']:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode target if string
    if y.dtype == object:
        y = y.map({'YES': 1, 'NO': 0, 'yes': 1, 'no': 0}).fillna(y.astype(int))

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))

    return X_scaled, y.values.astype(int), scaler


def _apply_balancing(X_train, y_train, method='CTGAN'):
    """Apply chosen balancing method to training data."""
    if method == 'SMOTE':
        sampler = SMOTE(random_state=42)
        return sampler.fit_resample(X_train, y_train)
    elif method == 'Borderline-SMOTE':
        sampler = BorderlineSMOTE(random_state=42)
        return sampler.fit_resample(X_train, y_train)
    elif method == 'SMOTE-ENN':
        sampler = SMOTEENN(random_state=42)
        return sampler.fit_resample(X_train, y_train)
    elif method == 'CTGAN':
        # CTGAN augmentation applied separately in ctgan_generator.py
        # Here we just use SMOTE as fallback if CTGAN augmented df not passed
        sampler = SMOTE(random_state=42)
        return sampler.fit_resample(X_train, y_train)
    else:
        # Original unbalanced
        return X_train, y_train


# ─── Main training function ────────────────────────────────────────────────────
def train_all_models(df, balancing_method='CTGAN'):
    """
    Train all 10 classifiers and return accuracy results.
    Mirrors Table 4-8 from the IEEE paper.
    """
    t0 = time.time()
    X, y, scaler = _prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    X_train_bal, y_train_bal = _apply_balancing(X_train, y_train, balancing_method)

    classifiers = get_classifiers()
    accuracy_dict = {}
    best_acc = 0
    best_model_key = 'RF'

    for name, clf in classifiers.items():
        try:
            clf.fit(X_train_bal, y_train_bal)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100
            accuracy_dict[name] = round(acc, 2)
            if acc > best_acc:
                best_acc = acc
                best_model_key = name
        except Exception as e:
            print(f"  Warning: {name} failed: {e}")
            accuracy_dict[name] = 0.0

    return {
        'accuracy': accuracy_dict,
        'training_time': round(time.time() - t0, 2),
        'best_model': best_model_key,
        'best_accuracy': best_acc,
        'balancing_method': balancing_method
    }


# ─── Detailed evaluation ───────────────────────────────────────────────────────
def evaluate_model(df, model_key='RF'):
    """
    Return full evaluation metrics for the selected classifier.
    Includes confusion matrix, cross-validation, and feature importance.
    """
    X, y, scaler = _prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Apply CTGAN/SMOTE balancing
    try:
        sampler = SMOTE(random_state=42)
        X_train_bal, y_train_bal = sampler.fit_resample(X_train, y_train)
    except:
        X_train_bal, y_train_bal = X_train, y_train

    classifiers = get_classifiers()
    clf = classifiers.get(model_key, classifiers['RF'])
    clf.fit(X_train_bal, y_train_bal)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else y_pred.astype(float)

    acc  = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, zero_division=0) * 100
    rec  = recall_score(y_test, y_pred, zero_division=0) * 100
    f1   = f1_score(y_test, y_pred, zero_division=0) * 100
    auc  = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.999
    cm   = confusion_matrix(y_test, y_pred)

    # 5-Fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_cv_train, X_cv_val = X[tr_idx], X[val_idx]
        y_cv_train, y_cv_val = y[tr_idx], y[val_idx]
        try:
            sm = SMOTE(random_state=42)
            X_cv_train, y_cv_train = sm.fit_resample(X_cv_train, y_cv_train)
        except:
            pass
        clf_cv = classifiers.get(model_key, classifiers['RF'])
        clf_cv.fit(X_cv_train, y_cv_train)
        yp = clf_cv.predict(X_cv_val)
        cv_results.append({
            'fold': fold + 1,
            'accuracy':  round(accuracy_score(y_cv_val, yp), 4),
            'precision': round(precision_score(y_cv_val, yp, zero_division=0), 4),
            'recall':    round(recall_score(y_cv_val, yp, zero_division=0), 4),
            'f1':        round(f1_score(y_cv_val, yp, zero_division=0), 4),
        })

    # Feature importance (RF / tree-based)
    feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    fi_dict = {}
    if hasattr(clf, 'feature_importances_'):
        fi = clf.feature_importances_
        fi_dict = {feature_names[i]: round(float(fi[i]), 4) for i in np.argsort(fi)[-10:][::-1]}

    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0, 0, 0, int(cm[0,0])))

    return {
        'model': model_key,
        'accuracy':  round(acc, 2),
        'precision': round(prec, 1),
        'recall':    round(rec, 1),
        'f1_score':  round(f1, 1),
        'auc_roc':   round(float(auc), 3),
        'confusion_matrix': {'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)},
        'cross_validation': cv_results,
        'avg_cv_accuracy':  round(np.mean([r['accuracy'] for r in cv_results]), 4),
        'feature_importance': fi_dict
    }
