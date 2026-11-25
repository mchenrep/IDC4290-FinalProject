"""Model library. Every model has the same small interface:

    m = Model(cat, num, seed=0, **params)
    m.fit(X_train, y_train, X_val, y_val)     # X_* are DataFrames; categorical columns have pandas 'category' dtype
    p = m.predict_proba(X)                    # 1-D probability of a severe crash

Models that cannot be imported (e.g. xgboost not installed) are simply left out of `available_models()`.
Baselines: dummy, logreg. The main models: rf, hist_gb, xgb, lgbm, catboost, nn.
"""
import importlib.util
import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


# ---------------------------------------------------------------- shared frame helpers
def make_category_frame(df, cat, num, levels=None):
    """Select columns; categorical columns become pandas 'category' with the SAME levels everywhere."""
    X = df[cat + num].copy()
    levels = levels or {}
    for c in cat:
        lv = levels.get(c)
        X[c] = pd.Categorical(X[c].astype(object), categories=lv) if lv is not None else X[c].astype("category")
    return X


def category_levels(X, cat):
    return {c: X[c].cat.categories for c in cat}


def make_frames(tr, va, te, cat, num):
    """Build train/val/test frames whose categorical levels come from the TRAINING data only."""
    X_tr = make_category_frame(tr, cat, num)
    levels = category_levels(X_tr, cat)
    return X_tr, make_category_frame(va, cat, num, levels), make_category_frame(te, cat, num, levels)


def to_str_frame(X, cat):
    X = X.copy()
    for c in cat:
        X[c] = X[c].astype(object).where(X[c].notna(), "UNKNOWN").astype(str)
    return X


class BaseModel:
    stochastic = True
    PARAM_SPACE = {}
    label = ""

    def __init__(self, cat, num, seed=0, **params):
        self.cat, self.num, self.seed, self.params = list(cat), list(num), seed, params

    def fit(self, X_tr, y_tr, X_va=None, y_va=None):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError


# ---------------------------------------------------------------- baselines
class DummyModel(BaseModel):
    stochastic = False
    label = "baseline (prior)"

    def fit(self, X_tr, y_tr, X_va=None, y_va=None):
        self.m = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)[:, 1]


class LogRegModel(BaseModel):
    stochastic = False
    label = "logistic regression"
    PARAM_SPACE = {"C": [0.1, 1.0, 10.0]}

    def fit(self, X_tr, y_tr, X_va=None, y_va=None):
        prep = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), self.cat),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), self.num)])
        self.m = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000, C=self.params.get("C", 1.0),
                                                                      class_weight="balanced"))])
        self.m.fit(to_str_frame(X_tr, self.cat), y_tr)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(to_str_frame(X, self.cat))[:, 1]


# ---------------------------------------------------------------- tree ensembles in scikit-learn
class RFModel(BaseModel):
    label = "random forest"
    PARAM_SPACE = {"min_samples_leaf": [5, 20, 50], "max_features": ["sqrt", 0.5], "max_depth": [None, 12, 20]}

    def fit(self, X_tr, y_tr, X_va=None, y_va=None):
        prep = ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), self.cat),
            ("num", SimpleImputer(strategy="median"), self.num)])
        rf = RandomForestClassifier(
            n_estimators=self.params.get("n_estimators", 200), min_samples_leaf=self.params.get("min_samples_leaf", 20),
            max_features=self.params.get("max_features", "sqrt"), max_depth=self.params.get("max_depth"),
            class_weight="balanced_subsample", n_jobs=-1, random_state=self.seed)
        self.m = Pipeline([("prep", prep), ("rf", rf)]).fit(to_str_frame(X_tr, self.cat), y_tr)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(to_str_frame(X, self.cat))[:, 1]


class HistGBModel(BaseModel):
    label = "gradient boosting (sklearn)"
    PARAM_SPACE = {"learning_rate": [0.03, 0.05, 0.1], "max_leaf_nodes": [15, 31, 63],
                   "min_samples_leaf": [20, 50, 100], "l2_regularization": [0.0, 1.0, 5.0]}

    def fit(self, X_tr, y_tr, X_va=None, y_va=None):
        p = self.params
        self.m = HistGradientBoostingClassifier(
            categorical_features="from_dtype", learning_rate=p.get("learning_rate", 0.05), max_iter=500,
            max_leaf_nodes=p.get("max_leaf_nodes", 31), min_samples_leaf=p.get("min_samples_leaf", 50),
            l2_regularization=p.get("l2_regularization", 1.0), early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=30, random_state=self.seed).fit(X_tr, y_tr)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)[:, 1]


# ---------------------------------------------------------------- optional libraries (skipped if not installed)
def _pos_weight(y, how):
    prev = float(np.mean(y))
    return math.sqrt((1 - prev) / prev) if how == "sqrt" else 1.0


class XGBModel(BaseModel):
    label = "XGBoost"
    PARAM_SPACE = {"max_depth": [4, 6, 8], "learning_rate": [0.03, 0.05, 0.1], "min_child_weight": [1, 5, 20],
                   "subsample": [0.7, 0.9, 1.0], "colsample_bytree": [0.7, 0.9, 1.0], "reg_lambda": [1.0, 5.0, 10.0],
                   "scale_pos_weight": ["none", "sqrt"]}

    def fit(self, X_tr, y_tr, X_va=None, y_va=None):
        from xgboost import XGBClassifier
        p = self.params
        self.m = XGBClassifier(
            n_estimators=1000, tree_method="hist", enable_categorical=True, eval_metric="aucpr",
            early_stopping_rounds=30, random_state=self.seed, n_jobs=-1, max_depth=p.get("max_depth", 6),
            learning_rate=p.get("learning_rate", 0.05), min_child_weight=p.get("min_child_weight", 5),
            subsample=p.get("subsample", 0.9), colsample_bytree=p.get("colsample_bytree", 0.9),
            reg_lambda=p.get("reg_lambda", 5.0), scale_pos_weight=_pos_weight(y_tr, p.get("scale_pos_weight", "none")))
        self.m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)[:, 1]


class LGBMModel(BaseModel):
    label = "LightGBM"
    PARAM_SPACE = {"num_leaves": [15, 31, 63], "learning_rate": [0.03, 0.05, 0.1], "min_child_samples": [20, 50, 100],
                   "subsample": [0.7, 0.9, 1.0], "colsample_bytree": [0.7, 0.9, 1.0], "reg_lambda": [0.0, 1.0, 5.0],
                   "scale_pos_weight": ["none", "sqrt"]}

    def fit(self, X_tr, y_tr, X_va=None, y_va=None):
        from lightgbm import LGBMClassifier, early_stopping
        p = self.params
        self.m = LGBMClassifier(
            n_estimators=1000, learning_rate=p.get("learning_rate", 0.05), num_leaves=p.get("num_leaves", 31),
            min_child_samples=p.get("min_child_samples", 50), subsample=p.get("subsample", 0.9), subsample_freq=1,
            colsample_bytree=p.get("colsample_bytree", 0.9), reg_lambda=p.get("reg_lambda", 1.0),
            scale_pos_weight=_pos_weight(y_tr, p.get("scale_pos_weight", "none")),
            random_state=self.seed, n_jobs=-1, verbose=-1)
        self.m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[early_stopping(30, verbose=False)])
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)[:, 1]


class CatBoostModel(BaseModel):
    label = "CatBoost"
    PARAM_SPACE = {"depth": [4, 6, 8], "learning_rate": [0.03, 0.05, 0.1], "l2_leaf_reg": [1.0, 3.0, 10.0],
                   "auto_class_weights": ["none", "SqrtBalanced"]}

    def fit(self, X_tr, y_tr, X_va=None, y_va=None):
        from catboost import CatBoostClassifier
        p = self.params
        acw = p.get("auto_class_weights", "none")
        self.m = CatBoostClassifier(
            iterations=1000, depth=p.get("depth", 6), learning_rate=p.get("learning_rate", 0.05),
            l2_leaf_reg=p.get("l2_leaf_reg", 3.0), auto_class_weights=None if acw == "none" else acw,
            early_stopping_rounds=30, random_seed=self.seed, verbose=0, allow_writing_files=False,
            cat_features=self.cat)
        self.m.fit(to_str_frame(X_tr, self.cat), y_tr, eval_set=(to_str_frame(X_va, self.cat), y_va))
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(to_str_frame(X, self.cat))[:, 1]


# ---------------------------------------------------------------- neural networks
class TorchMLPModel(BaseModel):
    """MLP with an embedding for every categorical column, BatchNorm + dropout, early stopping on validation PR-AUC."""
    label = "deep neural net (PyTorch MLP)"
    PARAM_SPACE = {"hidden": [(128, 64), (256, 128), (256, 128, 64)], "dropout": [0.1, 0.2, 0.3],
                   "lr": [1e-3, 3e-3], "weight_decay": [0.0, 1e-5, 1e-4], "pos_weight": ["none", "sqrt"]}
    MAX_EPOCHS, PATIENCE, BATCH = 25, 4, 1024

    def _encode(self, X):
        cat_idx = np.stack([pd.Series(X[c].astype(object).where(X[c].notna(), "UNKNOWN").astype(str)).map(self.levels[c])
                            .fillna(0).astype("int64").to_numpy() for c in self.cat], axis=1) if self.cat else \
            np.zeros((len(X), 0), dtype="int64")
        num = X[self.num].to_numpy(dtype="float32")
        num = np.where(np.isnan(num), self.med, num)
        return cat_idx, ((num - self.mu) / self.sd).astype("float32")

    def fit(self, X_tr, y_tr, X_va=None, y_va=None):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        p = self.params
        self.levels = {c: {v: i + 1 for i, v in enumerate(sorted(set(X_tr[c].astype(object).where(X_tr[c].notna(), "UNKNOWN").astype(str))))}
                       for c in self.cat}
        arr = X_tr[self.num].to_numpy(dtype="float32")
        self.med = np.nanmedian(arr, axis=0)
        filled = np.where(np.isnan(arr), self.med, arr)
        self.mu, self.sd = filled.mean(axis=0), filled.std(axis=0) + 1e-6

        class Net(nn.Module):
            def __init__(s, cards, n_num, hidden, dropout):
                super().__init__()
                s.embs = nn.ModuleList([nn.Embedding(k + 1, min(50, (k + 2) // 2)) for k in cards])
                d = sum(e.embedding_dim for e in s.embs) + n_num
                layers = []
                for h in hidden:
                    layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
                    d = h
                layers.append(nn.Linear(d, 1))
                s.mlp = nn.Sequential(*layers)

            def forward(s, cat_idx, num):
                parts = [e(cat_idx[:, i]) for i, e in enumerate(s.embs)] + [num]
                return s.mlp(torch.cat(parts, dim=1)).squeeze(1)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = Net([len(self.levels[c]) for c in self.cat], len(self.num), tuple(p.get("hidden", (128, 64))),
                       p.get("dropout", 0.2)).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=p.get("lr", 1e-3), weight_decay=p.get("weight_decay", 0.0))
        pw = torch.tensor(_pos_weight(y_tr, p.get("pos_weight", "none")), dtype=torch.float32, device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        c_tr, n_tr = self._encode(X_tr)
        y_arr = np.asarray(y_tr, dtype="float32")
        best, best_state, bad = -1.0, None, 0
        for _ in range(self.MAX_EPOCHS):
            self.net.train()
            perm = np.random.permutation(len(y_arr))
            for i in range(0, len(perm), self.BATCH):
                b = perm[i:i + self.BATCH]
                if len(b) < 2:
                    continue
                opt.zero_grad()
                logits = self.net(torch.from_numpy(c_tr[b]).to(self.device), torch.from_numpy(n_tr[b]).to(self.device))
                loss_fn(logits, torch.from_numpy(y_arr[b]).to(self.device)).backward()
                opt.step()
            score = average_precision_score(y_va, self.predict_proba(X_va)) if X_va is not None else 0.0
            if score > best:
                best, bad = score, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.PATIENCE:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        import torch
        self.net.eval()
        c, n = self._encode(X)
        out = []
        with torch.no_grad():
            for i in range(0, len(n), 8192):
                logits = self.net(torch.from_numpy(c[i:i + 8192]).to(self.device), torch.from_numpy(n[i:i + 8192]).to(self.device))
                out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out)


class SkMLPModel(BaseModel):
    """Fallback 'deep net' when PyTorch is not installed: scikit-learn MLP on one-hot + scaled inputs."""
    label = "neural net (sklearn MLP, PyTorch not installed)"
    PARAM_SPACE = {"hidden": [(64,), (128, 64)], "alpha": [1e-5, 1e-4, 1e-3]}

    def fit(self, X_tr, y_tr, X_va=None, y_va=None):
        prep = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), self.cat),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), self.num)])
        mlp = MLPClassifier(hidden_layer_sizes=tuple(self.params.get("hidden", (128, 64))), alpha=self.params.get("alpha", 1e-4),
                            early_stopping=True, n_iter_no_change=5, max_iter=60, random_state=self.seed)
        self.m = Pipeline([("prep", prep), ("mlp", mlp)]).fit(to_str_frame(X_tr, self.cat), y_tr)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(to_str_frame(X, self.cat))[:, 1]


def _has(pkg):
    return importlib.util.find_spec(pkg) is not None


def available_models():
    """name -> model class. Libraries that are not installed are skipped (with a printed note by the caller)."""
    models = {"dummy": DummyModel, "logreg": LogRegModel, "rf": RFModel, "hist_gb": HistGBModel}
    if _has("xgboost"):
        models["xgb"] = XGBModel
    if _has("lightgbm"):
        models["lgbm"] = LGBMModel
    if _has("catboost"):
        models["catboost"] = CatBoostModel
    models["nn"] = TorchMLPModel if _has("torch") else SkMLPModel
    return models


BASELINES = {"dummy", "logreg"}
