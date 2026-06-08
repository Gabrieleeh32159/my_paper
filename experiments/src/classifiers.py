"""
Classifier wrappers for TSI experiments.

Implements: Random Forest, XGBoost, SVM (RBF), MLP.
All classifiers follow a common interface for easy swapping.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier as _SklearnRF
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Optional

# torch / xgboost are imported lazily so the analysis modules and tests can import
# the lightweight classifiers (RF, SVM, multilabel wrapper) without them. In Colab
# both are installed by the notebook's setup cell.
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch present in the Colab runtime
    torch = None
    nn = None
    _HAS_TORCH = False

# Dimensionality of the early-fusion representation (3 temporal scales x 192-d per
# scale). The paper restricts SVM's 50% MTAT subsample to THIS representation
# only, so :class:`SVMClassifier` keys the subsample decision on it.
EARLY_FUSION_DIM = 576


def _make_rf_estimator(n_estimators: int, max_depth: int, random_state: int):
    """Return a cuML RandomForestClassifier if available, else sklearn's.

    cuML RF runs fully on CUDA (A100 etc.) and is typically 10-50x faster than
    sklearn's CPU implementation. The wrapper below converts cupy outputs to
    numpy so CalibratedClassifierCV and the rest of the sklearn pipeline work
    transparently.
    """
    try:
        from cuml.ensemble import RandomForestClassifier as _CuRF

        class _CuMLRFWrapper(BaseEstimator, ClassifierMixin):
            """Thin sklearn-compatible wrapper around cuML's RandomForestClassifier."""

            def __init__(self, n_estimators=100, max_depth=30, random_state=42):
                self.n_estimators = n_estimators
                self.max_depth = max_depth
                self.random_state = random_state

            def fit(self, X, y):
                self._rf = _CuRF(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                )
                self._rf.fit(X, y)
                self.classes_ = np.unique(y)
                return self

            def predict(self, X):
                return np.asarray(self._rf.predict(X)).astype(int)

            def predict_proba(self, X):
                return np.asarray(self._rf.predict_proba(X))

            @property
            def feature_importances_(self):
                return np.asarray(self._rf.feature_importances_)

        return _CuMLRFWrapper(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

    except ImportError:
        return _SklearnRF(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )


def _calibrated_base_estimators(calibrated_clf):
    """Return the list of fitted base estimators inside a CalibratedClassifierCV.

    Handles sklearn naming changes ('estimator' vs. legacy 'base_estimator').
    """
    estimators = []
    for cc in calibrated_clf.calibrated_classifiers_:
        est = getattr(cc, 'estimator', None)
        if est is None:
            est = getattr(cc, 'base_estimator', None)
        if est is not None:
            estimators.append(est)
    return estimators


class RFClassifier:
    """Random Forest classifier (500 trees, max_depth=30).

    Probabilities are calibrated by default (``CalibratedClassifierCV``) so that
    ``predict_proba`` is suitable for log-loss / information-gain based metrics
    such as the TSI. Set ``calibrate=False`` to recover raw RF probabilities.
    """

    def __init__(self, n_estimators: int = 500, max_depth: int = 30,
                 random_state: int = 42, calibrate: bool = True,
                 calibration_method: str = 'sigmoid', calibration_cv: int = 3):
        self.base = _make_rf_estimator(n_estimators, max_depth, random_state)
        # which backend was actually selected, for reproducibility reporting
        self.backend = 'cuml' if type(self.base).__name__ == '_CuMLRFWrapper' else 'sklearn'
        self.hyperparams = {'n_estimators': n_estimators, 'max_depth': max_depth,
                            'backend': self.backend}
        self.calibrate = calibrate
        if calibrate:
            self.model = CalibratedClassifierCV(
                self.base, method=calibration_method, cv=calibration_cv
            )
        else:
            self.model = self.base
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    @property
    def feature_importances_(self):
        # When calibrated, average MDI across the per-fold base estimators.
        if self.calibrate:
            imps = [est.feature_importances_
                    for est in _calibrated_base_estimators(self.model)]
            return np.mean(imps, axis=0)
        return self.model.feature_importances_


class XGBoostClassifier:
    """XGBoost classifier with sensible defaults.

    Like :class:`RFClassifier`, probabilities are calibrated by default so the
    log-loss / information-gain TSI is computed on trustworthy probabilities.
    """

    def __init__(self, n_estimators: int = 500, max_depth: int = 6,
                 learning_rate: float = 0.1, random_state: int = 42,
                 calibrate: bool = True, calibration_method: str = 'sigmoid',
                 calibration_cv: int = 3, device: Optional[str] = None):
        from xgboost import XGBClassifier as _XGBClassifier
        # ``device=None`` -> auto-detect (GPU if torch sees one). Pass ``'cpu'``
        # explicitly to force the CPU ``hist`` method: for the tiny per-descriptor
        # fits in the TSI sweep the GPU's per-round kernel-launch overhead usually
        # makes CPU far faster. ``device`` is NOT part of any checkpoint signature,
        # so switching it never invalidates cached gain records.
        if device is None:
            try:
                import torch as _torch
                device = 'cuda' if _torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'
        _device = device
        self.base = _XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1 if _device == 'cpu' else 1,
            tree_method='hist',
            device=_device,
            eval_metric='mlogloss',
            use_label_encoder=False,
            verbosity=0,
        )
        # recorded for reproducibility (the paper does not fix these)
        self.hyperparams = {'n_estimators': n_estimators, 'max_depth': max_depth,
                            'learning_rate': learning_rate, 'device': _device,
                            'calibration': f'{calibration_method}/cv{calibration_cv}'}
        self.calibrate = calibrate
        if calibrate:
            self.model = CalibratedClassifierCV(
                self.base, method=calibration_method, cv=calibration_cv
            )
        else:
            self.model = self.base
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    @property
    def feature_importances_(self):
        if self.calibrate:
            imps = [est.feature_importances_
                    for est in _calibrated_base_estimators(self.model)]
            return np.mean(imps, axis=0)
        return self.model.feature_importances_


class SVMClassifier:
    """SVM with RBF kernel, hyperparameters tuned via 3-fold CV.

    ``predict_proba`` uses libsvm's built-in Platt scaling (``probability=True``),
    which already calibrates the probabilities, so no extra calibration wrapper
    is applied here.

    The optional ``subsample`` knob (wired by :func:`make_clf_factory` for MTAT)
    applies the paper's 50% random training subsample **only** to SVM's 576-d
    early-fusion representation. Lower-dimensional fits — the single-descriptor TSI
    sweeps (``4xdim``) and the single-scale 192-d representation — always train on
    the full per-fold training set; see :meth:`_should_subsample`.
    """

    def __init__(self, random_state: int = 42, subsample: Optional[float] = None):
        self.random_state = random_state
        self.subsample = subsample
        self.scaler = StandardScaler()
        self.model = None

    def _should_subsample(self, X: np.ndarray) -> bool:
        """Whether to apply the 50% training subsample to this representation.

        Paper protocol: the subsample is restricted to the 576-d early-fusion
        representation on MTAT. It therefore fires only when the ``subsample``
        knob is set AND the input dimensionality equals
        :data:`EARLY_FUSION_DIM` (576). Single-descriptor TSI fits (``4xdim``) and
        the single-scale 192-d representation are never subsampled.
        """
        return bool(self.subsample) and X.shape[1] == EARLY_FUSION_DIM

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Fit scaler on full training data first (consistent statistics)
        X_scaled = self.scaler.fit_transform(X)

        # 50% subsample only for the 576-d early-fusion representation (MTAT);
        # only ever touches this per-fold training set, with a fixed seed.
        if self._should_subsample(X_scaled):
            rng = np.random.RandomState(self.random_state)
            n_sub = int(len(X) * self.subsample)
            idx = rng.choice(len(X), n_sub, replace=False)
            X_fit, y_fit = X_scaled[idx], y[idx]
        else:
            X_fit, y_fit = X_scaled, y

        # Grid search for C and gamma
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.001]
        }

        svm = SVC(kernel='rbf', probability=True, random_state=self.random_state)
        grid = GridSearchCV(
            svm, param_grid, cv=3, scoring='f1_macro',
            n_jobs=-1, refit=True
        )
        grid.fit(X_fit, y_fit)
        self.model = grid.best_estimator_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


def inverse_frequency_pos_weight(
    y_train: np.ndarray, clip_lo: float = 1e-3, clip_hi: float = 1e3
) -> np.ndarray:
    """Per-tag BCE ``pos_weight`` inversely proportional to tag frequency.

    For each tag ``t``, ``pos_weight_t = n_neg_t / n_pos_t`` computed from the
    *training* labels only (paper, Sec. Datasets / Modelos: MTAT MLP uses weighted
    binary cross-entropy with weights inverse to each tag's frequency). Clipped to
    ``[clip_lo, clip_hi]`` so a tag absent (``n_pos=0``) or saturated (``n_neg=0``)
    in this fold yields a finite, well-behaved weight instead of ``inf``/``0``.

    Parameters
    ----------
    y_train : np.ndarray
        Binary tag matrix of the training fold, shape ``(n, T)`` (a 1-D vector is
        treated as a single tag).

    Returns
    -------
    np.ndarray
        ``float32`` vector of length ``T`` with each entry in ``[clip_lo, clip_hi]``.
    """
    y = np.asarray(y_train, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    n = y.shape[0]
    n_pos = y.sum(axis=0)
    n_neg = n - n_pos
    # n_pos==0 -> saturate to clip_hi (rare tag); else n_neg/n_pos.
    pw = np.where(n_pos > 0, n_neg / np.maximum(n_pos, 1e-12), clip_hi)
    return np.clip(pw, clip_lo, clip_hi).astype(np.float32)


# Base is nn.Module when torch is available, else a stub so the module still
# imports; instantiating MLPModel without torch raises a clear error.
_MLPBase = nn.Module if _HAS_TORCH else object


class MLPModel(_MLPBase):
    """MLP: 2 hidden layers (256, 128), ReLU, dropout=0.3."""

    def __init__(self, input_dim: int, n_classes: int, dropout: float = 0.3):
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for the MLP classifier.")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.net(x)


class MLPClassifier:
    """
    MLP classifier with early stopping.

    For multiclass: cross-entropy loss.
    For multilabel: weighted binary cross-entropy.
    """

    def __init__(self, n_classes: int, task_type: str = 'multiclass',
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 epochs: int = 100, patience: int = 10,
                 batch_size: int = 256, random_state: int = 42,
                 class_weights: Optional[np.ndarray] = None,
                 auto_class_weights: bool = False):
        self.n_classes = n_classes
        self.task_type = task_type
        self.lr = lr
        self.weight_decay = weight_decay
        # When True (and the task is multilabel), derive per-tag inverse-frequency
        # BCE pos_weights from each fold's training labels inside ``fit`` (paper's
        # weighted BCE for MTAT). Wired by ``make_clf_factory`` for MTAT+MLP.
        self.auto_class_weights = auto_class_weights
        self.hyperparams = {'hidden': (256, 128), 'dropout': 0.3, 'lr': lr,
                            'weight_decay': weight_decay, 'epochs': epochs,
                            'patience': patience, 'batch_size': batch_size,
                            'calibration': 'temperature_scaling',
                            'val_split': 0.15,
                            'auto_class_weights': auto_class_weights}
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.random_state = random_state
        self.class_weights = class_weights
        self.scaler = StandardScaler()
        self.model = None
        # Temperature scaling factor for probability calibration. Fitted on the
        # validation set when one is provided to ``fit``; otherwise left at 1.0
        # (softmax/sigmoid of a cross-entropy-trained net is already roughly
        # calibrated).
        self.temperature = 1.0
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

    def _fit_temperature(self, logits: "torch.Tensor", targets: "torch.Tensor"):
        """Optimize a single temperature scalar to minimize NLL on a held-out set."""
        T = torch.ones(1, device=self.device, requires_grad=True)
        optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=50)

        if self.task_type == 'multilabel':
            criterion = nn.BCEWithLogitsLoss()
            tgt = targets.float()
        else:
            criterion = nn.CrossEntropyLoss()
            tgt = targets.long()

        def closure():
            optimizer.zero_grad()
            loss = criterion(logits / T.clamp_min(1e-3), tgt)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(T.detach().clamp_min(1e-3).item())

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # When no validation set is supplied (the generic clf_factory.fit(X, y)
        # path used by the TSI/fusion drivers), carve one out internally so the
        # MLP still gets early stopping AND temperature scaling — otherwise the
        # temperature stays 1.0 and the probabilities are left uncalibrated.
        if X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split
            X = np.asarray(X)
            y = np.asarray(y)
            strat = y if self.task_type != 'multilabel' else None
            if len(y) >= 20:
                try:
                    X, X_val, y, y_val = train_test_split(
                        X, y, test_size=0.15,
                        random_state=self.random_state, stratify=strat)
                except ValueError:
                    # too few samples per class to stratify -> split without it
                    X, X_val, y, y_val = train_test_split(
                        X, y, test_size=0.15, random_state=self.random_state)

        X_scaled = self.scaler.fit_transform(X)
        input_dim = X_scaled.shape[1]

        self.model = MLPModel(input_dim, self.n_classes).to(self.device)
        if hasattr(torch, 'compile') and self.device.type == 'cuda':
            self.model = torch.compile(self.model, mode='reduce-overhead')
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Weighted BCE for multilabel: derive per-tag inverse-frequency pos_weights
        # from THIS fold's training labels only (the internal val carved out above
        # is excluded; val/test are never touched). Only when explicitly enabled
        # (MTAT+MLP via make_clf_factory) and not overridden by explicit weights.
        if (self.task_type == 'multilabel' and self.auto_class_weights
                and self.class_weights is None):
            self.class_weights = inverse_frequency_pos_weight(y)

        # Loss function
        if self.task_type == 'multilabel':
            if self.class_weights is not None:
                pos_weight = torch.tensor(self.class_weights, dtype=torch.float32).to(self.device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        # DataLoader
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        if self.task_type == 'multilabel':
            y_tensor = torch.tensor(y, dtype=torch.float32)
        else:
            y_tensor = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            pin_memory=(self.device.type != 'cpu'),
            num_workers=4,
            persistent_workers=True,
        )

        # Validation set
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
            if self.task_type == 'multilabel':
                y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)
            else:
                y_val_t = torch.tensor(y_val, dtype=torch.long).to(self.device)

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

            # Validation
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_val_t)
                    val_loss = criterion(val_logits, y_val_t).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        self.model.eval()

        # Calibrate probabilities via temperature scaling on the validation set.
        if has_val:
            with torch.no_grad():
                val_logits = self.model(X_val_t)
            self._fit_temperature(val_logits, y_val_t)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(X_tensor)

        if self.task_type == 'multilabel':
            return (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(int)
        else:
            return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(X_tensor) / self.temperature

        if self.task_type == 'multilabel':
            return torch.sigmoid(logits).cpu().numpy()
        else:
            return torch.softmax(logits, dim=1).cpu().numpy()


class MultiLabelWrapper:
    """Per-tag binary wrapper turning a multiclass classifier into a multilabel one.

    Trains one independent binary (calibrated) classifier per tag and stacks the
    positive-class probabilities into an ``(n, T)`` matrix. This lets the
    tree-based classifiers (XGBoost/RF) -- which are single-output -- be used on
    MTAT's 50-tag multilabel task. The MLP already handles multilabel natively, so
    it is not wrapped.

    NOTE: training ``T`` calibrated classifiers is expensive; the notebook scopes
    this (e.g. SVM/MTAT subsampling) and XGBoost/RF remain the practical choices.
    """

    def __init__(self, base_factory, n_tags: int):
        self.base_factory = base_factory
        self.n_tags = n_tags
        self.models = []

    def fit(self, X, y):
        y = np.asarray(y)
        self.models = []
        for t in range(self.n_tags):
            yt = y[:, t].astype(int)
            clf = self.base_factory()
            if len(np.unique(yt)) < 2:
                # degenerate tag in this fold: remember the constant prevalence,
                # clipped off 0/1 exactly like evaluation.base_rate so it stays
                # consistent with L_chance (a 0/1 constant would diverge in BCE).
                from .evaluation import PREVALENCE_EPS
                prevalence = float(np.clip(yt.mean(), PREVALENCE_EPS, 1.0 - PREVALENCE_EPS))
                self.models.append(('const', prevalence))
            else:
                clf.fit(X, yt)
                self.models.append(('clf', clf))
        return self

    def predict_proba(self, X):
        cols = []
        for kind, obj in self.models:
            if kind == 'const':
                cols.append(np.full(X.shape[0], obj))
            else:
                p = obj.predict_proba(X)
                # positive-class probability
                cols.append(p[:, 1] if p.shape[1] > 1 else p[:, 0])
        return np.stack(cols, axis=1)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def get_classifier(name: str, input_dim: int, n_classes: int,
                   task_type: str = 'multiclass', **kwargs):
    """
    Factory function to create a classifier by name.

    Parameters
    ----------
    name : str
        One of 'rf', 'xgb', 'svm', 'mlp'.
    input_dim : int
        Number of input features.
    n_classes : int
        Number of output classes (tags for multilabel).
    task_type : str
        'multiclass' or 'multilabel'.
    **kwargs :
        Extra classifier-specific keyword arguments forwarded verbatim to the
        underlying wrapper (e.g. ``subsample=0.5`` for :class:`SVMClassifier` on
        MTAT). For multilabel tasks they reach the per-tag base classifier
        through :class:`MultiLabelWrapper`'s factory.

    Returns
    -------
    Classifier instance with fit/predict/predict_proba interface. For multilabel
    tasks, tree/SVM classifiers are wrapped per-tag (:class:`MultiLabelWrapper`);
    the MLP handles multilabel natively.
    """
    if name == 'mlp':
        return MLPClassifier(n_classes=n_classes, task_type=task_type, **kwargs)

    base_factories = {
        'rf': lambda: RFClassifier(**kwargs),
        'xgb': lambda: XGBoostClassifier(**kwargs),
        'svm': lambda: SVMClassifier(**kwargs),
    }
    if name not in base_factories:
        raise ValueError(f"Unknown classifier: {name}. Choose from ['rf', 'xgb', 'svm', 'mlp']")

    if task_type == 'multilabel':
        return MultiLabelWrapper(base_factories[name], n_tags=n_classes)
    return base_factories[name]()


def make_clf_factory(clf_name: str, dataset: Optional[str] = None, **extra):
    """Build a classifier factory closure for the TSI / fusion drivers.

    The drivers call the returned ``factory(input_dim, n_classes, task_type)``
    once per fold/representation. This is where dataset-specific protocol knobs
    from the paper are wired in so they actually reach the classifier:

    - **SVM on MTAT** trains its 576-d **early-fusion** representation with a 50%
      random subsample of the training tracks (the paper restricts the subsample
      to this representation). The knob is passed to every SVM+MTAT fit, but
      :meth:`SVMClassifier.fit` only acts on it when the input dimensionality is
      :data:`EARLY_FUSION_DIM` (576); the lower-dimensional single-descriptor TSI
      (``4xdim``) and single-scale (192-d) fits use the full per-fold training
      set. The subsample only ever touches the per-fold *training* set (never
      val/test) and uses a fixed seed.
    - **MLP on MTAT** uses weighted binary cross-entropy with per-tag
      inverse-frequency ``pos_weight`` (``auto_class_weights=True``); the weights
      are computed per fold from the *training* labels inside
      :meth:`MLPClassifier.fit`. Multiclass tasks keep standard cross-entropy.

    Any ``extra`` keyword arguments are forwarded verbatim to
    :func:`get_classifier` (and override the dataset-derived defaults).
    """
    kwargs = dict(extra)
    if clf_name == 'svm' and dataset == 'mtat':
        kwargs.setdefault('subsample', 0.5)
    # MTAT + MLP: weighted binary cross-entropy with per-tag inverse-frequency
    # weights (paper, Sec. Datasets / Modelos). The weights are computed per fold
    # from the training labels inside MLPClassifier.fit (see auto_class_weights).
    if clf_name == 'mlp' and dataset == 'mtat':
        kwargs.setdefault('auto_class_weights', True)

    def factory(input_dim, n_classes, task_type):
        return get_classifier(clf_name, input_dim, n_classes, task_type, **kwargs)

    return factory
