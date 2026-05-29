# -*- coding: utf-8 -*-
"""Production-safe preprocessor for the AI Student Impact (Tükenmişlik) classifier.

Tek-tıkla deployment için tasarlandı: ham Gradio formundan gelen DataFrame
doğrudan ``ProductionPreprocessor.transform`` ile modele hazır hale gelir.
Tüm öğrenilmiş durum yalnızca ``X_train`` üstünde fit edilir; test ve canlı
girdi üstünde sadece transform çağrılır. Bu modül notebook tarafından da,
joblib pickle yüklemeleri tarafından da import edilir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from category_encoders import WOEEncoder
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import OrdinalEncoder, RobustScaler


LR_NUMERIC = ['Haftalık AI Saati', 'Algılanan AI Bağımlılığı']
KNN_NUMERIC = ['Sınav Kaygı Düzeyi', 'Geleneksel Çalışma Saati', 'Beceri Kalıcılık Skoru']
MED_NUMERIC = ['Araç Çeşitliliği']
ALL_NUMERIC = LR_NUMERIC + KNN_NUMERIC + MED_NUMERIC

ORDINAL_COLS = ['Sınıf Düzeyi', 'Prompt Yazma Becerisi', 'Kurum Politikası']
ORDINAL_ORDERS = [
    ['1. Sınıf', '2. Sınıf', '3. Sınıf', '4. Sınıf', 'Lisansüstü'],
    ['Başlangıç', 'Orta', 'İleri'],
    ['Kesin Yasak', 'Atıfla İzinli', 'Aktif Teşvik'],
]

WOE_COLS = ['Okunan Bölüm', 'Birincil Kullanım Amacı']
BOOL_COLS = ['Ücretli Abonelik']

DOMAIN_CLIP_COLS = ['Haftalık AI Saati', 'Geleneksel Çalışma Saati']
WEAK_MI_THRESHOLD = 0.005


class ProductionPreprocessor:
    """Ham girdi → modele hazır sütunlar (yalnızca train üstünde fit)."""

    feature_names_ = ALL_NUMERIC + ORDINAL_COLS + WOE_COLS + BOOL_COLS

    def __init__(self, weak_mi_threshold: float = WEAK_MI_THRESHOLD):
        self.weak_mi_threshold = weak_mi_threshold

        self.numeric_imputer = ColumnTransformer([
            ('lr_imp',  IterativeImputer(estimator=BayesianRidge(), random_state=42, max_iter=10), LR_NUMERIC),
            ('knn_imp', KNNImputer(n_neighbors=5, weights='distance'),                              KNN_NUMERIC),
            ('med_imp', SimpleImputer(strategy='median'),                                           MED_NUMERIC),
        ], remainder='drop')
        self.scaler = RobustScaler()

        self.ordinal_imputer = SimpleImputer(strategy='most_frequent')
        self.ordinal_encoder = OrdinalEncoder(
            categories=ORDINAL_ORDERS,
            handle_unknown='use_encoded_value', unknown_value=-1,
        )

        self.woe_imputer = SimpleImputer(strategy='most_frequent')
        self.woe_encoder = WOEEncoder(cols=WOE_COLS, randomized=False)

        self.bool_imputer = SimpleImputer(strategy='most_frequent')

        self.selected_features_: list[str] | None = None
        self.dropped_weak_: list[str] | None = None
        self.mi_scores_: pd.Series | None = None

    @staticmethod
    def _coerce_bool(frame: pd.DataFrame) -> pd.DataFrame:
        """Bool/Int64/string → float, NaN korunur. SimpleImputer numeric ister."""
        out = frame.copy()
        for col in out.columns:
            s = out[col]
            if s.dtype == 'bool':
                out[col] = s.astype(float)
            elif str(s.dtype) == 'boolean' or str(s.dtype) == 'Int64':
                out[col] = s.astype('Float64').astype(float)
            else:
                out[col] = pd.to_numeric(s, errors='coerce')
        return out

    def _clip_domain(self, arr) -> np.ndarray:
        df = pd.DataFrame(arr, columns=ALL_NUMERIC).copy()
        for col in DOMAIN_CLIP_COLS:
            df[col] = df[col].clip(lower=0)
        return df.values

    def fit(self, X: pd.DataFrame, y_multiclass, y_binary_for_woe):
        num_imputed = self.numeric_imputer.fit_transform(X[ALL_NUMERIC])
        num_clipped = self._clip_domain(num_imputed)
        self.scaler.fit(num_clipped)

        ord_imputed = self.ordinal_imputer.fit_transform(X[ORDINAL_COLS])
        self.ordinal_encoder.fit(ord_imputed)

        woe_imputed = self.woe_imputer.fit_transform(X[WOE_COLS])
        woe_df = pd.DataFrame(woe_imputed, columns=WOE_COLS, index=X.index)
        self.woe_encoder.fit(woe_df, y_binary_for_woe)

        self.bool_imputer.fit(self._coerce_bool(X[BOOL_COLS]))

        full_df = self._transform_to_df(X)
        self.mi_scores_ = pd.Series(
            mutual_info_classif(full_df.values, y_multiclass, random_state=42),
            index=self.feature_names_,
        ).sort_values(ascending=False)
        self.selected_features_ = self.mi_scores_[self.mi_scores_ >= self.weak_mi_threshold].index.tolist()
        self.dropped_weak_ = self.mi_scores_[self.mi_scores_ < self.weak_mi_threshold].index.tolist()
        return self

    def _transform_to_df(self, X: pd.DataFrame) -> pd.DataFrame:
        num_imputed = self.numeric_imputer.transform(X[ALL_NUMERIC])
        num_clipped = self._clip_domain(num_imputed)
        num_scaled = self.scaler.transform(num_clipped)

        ord_imputed = self.ordinal_imputer.transform(X[ORDINAL_COLS])
        ord_enc = self.ordinal_encoder.transform(ord_imputed)

        woe_imputed = self.woe_imputer.transform(X[WOE_COLS])
        woe_df = pd.DataFrame(woe_imputed, columns=WOE_COLS, index=X.index)
        woe_enc = self.woe_encoder.transform(woe_df).values

        bool_imputed = self.bool_imputer.transform(self._coerce_bool(X[BOOL_COLS]))
        bool_arr = pd.DataFrame(bool_imputed, columns=BOOL_COLS).astype(float).values

        full = np.hstack([num_scaled, ord_enc, woe_enc, bool_arr])
        return pd.DataFrame(full, columns=self.feature_names_, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = self._transform_to_df(X)
        if self.selected_features_ is None:
            return df
        return df[self.selected_features_]
