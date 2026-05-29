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
from xgboost import XGBClassifier as _XGBClassifier


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

            def __init__(self, n_estimators=100, max_depth=16, random_state=42):
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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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
                 calibration_cv: int = 3):
        import torch as _torch
        _device = 'cuda' if _torch.cuda.is_available() else 'cpu'
        self.base = _XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1 if _device == 'cpu' else 1,
            device=_device,
            eval_metric='mlogloss',
            use_label_encoder=False,
            verbosity=0,
        )
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
    """

    def __init__(self, random_state: int = 42, subsample: Optional[float] = None):
        self.random_state = random_state
        self.subsample = subsample
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Fit scaler on full training data first (consistent statistics)
        X_scaled = self.scaler.fit_transform(X)

        # Subsample for large datasets (e.g., MTAT)
        if self.subsample and len(X) > 5000:
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


class MLPModel(nn.Module):
    """MLP: 2 hidden layers (256, 128), ReLU, dropout=0.3."""

    def __init__(self, input_dim: int, n_classes: int, dropout: float = 0.3):
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
                 class_weights: Optional[np.ndarray] = None):
        self.n_classes = n_classes
        self.task_type = task_type
        self.lr = lr
        self.weight_decay = weight_decay
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

    def _fit_temperature(self, logits: torch.Tensor, targets: torch.Tensor):
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

        X_scaled = self.scaler.fit_transform(X)
        input_dim = X_scaled.shape[1]

        self.model = MLPModel(input_dim, self.n_classes).to(self.device)
        if hasattr(torch, 'compile') and self.device.type == 'cuda':
            self.model = torch.compile(self.model, mode='reduce-overhead')
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

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


def get_classifier(name: str, input_dim: int, n_classes: int,
                   task_type: str = 'multiclass', **kwargs):
    """
    Factory function to create a classifier by name.

    Parameters
    ----------
    name : str
        One of 'rf', 'svm', 'mlp'.
    input_dim : int
        Number of input features.
    n_classes : int
        Number of output classes.
    task_type : str
        'multiclass' or 'multilabel'.

    Returns
    -------
    Classifier instance with fit/predict/predict_proba interface.
    """
    if name == 'rf':
        return RFClassifier(**kwargs)
    elif name == 'xgb':
        return XGBoostClassifier(**kwargs)
    elif name == 'svm':
        return SVMClassifier(**kwargs)
    elif name == 'mlp':
        return MLPClassifier(n_classes=n_classes, task_type=task_type, **kwargs)
    else:
        raise ValueError(f"Unknown classifier: {name}. Choose from ['rf', 'xgb', 'svm', 'mlp']")
