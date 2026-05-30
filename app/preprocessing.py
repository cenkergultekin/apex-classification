# -*- coding: utf-8 -*-
"""Production-safe preprocessor for the AI Student Impact classifier.

Ham formdan gelen DataFrame doğrudan ``ProductionPreprocessor.transform`` ile
modele hazır hale gelir. Öğrenilen tüm eşikler/imputer/encoder/scaler yalnızca
``fit`` sırasında verilen train verisinden öğrenilir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler


BASE_NUMERIC = [
    'Dönem Öncesi GNO',
    'Dönem Sonrası GNO',
    'Haftalık AI Saati',
    'Algılanan AI Bağımlılığı',
    'Sınav Kaygı Düzeyi',
    'Geleneksel Çalışma Saati',
    'Beceri Kalıcılık Skoru',
    'Araç Çeşitliliği',
]

ENGINEERED_NUMERIC = [
    'GNO Değişimi',
    'GNO Ortalaması',
    'GNO Düştü Mü',
    'GNO Düşüş Şiddeti',
    'AI / Çalışma Oranı',
    'AI Çalışma Payı',
    'Çalışma Dengesizliği',
    'AI Bağımlılık Yükü',
    'Sınav Kaygısı × AI Saati',
    'AI × Kaygı × Bağımlılık',
    'Kalıcılık / AI Saati',
    'Kaygı / Kalıcılık',
    'Bağımlılık / Kalıcılık',
    'AI × Düşük Kalıcılık',
    'Toplam Çalışma Yükü',
    'Yüksek AI Kullanımı',
    'Yüksek Bağımlılık',
    'Yüksek Kaygı',
    'Düşük Kalıcılık',
    'Yüksek AI + Yüksek Kaygı',
    'Yüksek AI + Düşük Kalıcılık',
    'Yüksek AI + Yüksek Bağımlılık',
    'Risk Üçlüsü Bayrağı',
    'Koruyucu Profil Bayrağı',
]

ALL_NUMERIC = BASE_NUMERIC + ENGINEERED_NUMERIC

ORDINAL_COLS = ['Sınıf Düzeyi', 'Prompt Yazma Becerisi', 'Kurum Politikası']
ORDINAL_ORDERS = [
    ['1. Sınıf', '2. Sınıf', '3. Sınıf', '4. Sınıf', 'Yüksek Lisans'],
    ['Başlangıç', 'Orta', 'İleri'],
    ['Kesin Yasak', 'Kaynak Belirterek İzinli', 'Aktif Olarak Teşvik'],
]

OHE_COLS = ['Okunan Bölüm', 'Birincil Kullanım Amacı']
BOOL_COLS = ['Ücretli Abonelik']

MISSING_INDICATOR_COLS = BASE_NUMERIC + ['Prompt Yazma Becerisi']

COMPACT_CORE_FEATURES = [
    'Haftalık AI Saati',
    'Algılanan AI Bağımlılığı',
    'AI Bağımlılık Yükü',
    'Sınav Kaygısı × AI Saati',
    'AI × Kaygı × Bağımlılık',
    'AI Çalışma Payı',
    'Kalıcılık / AI Saati',
    'AI × Düşük Kalıcılık',
    'Bağımlılık / Kalıcılık',
    'Toplam Çalışma Yükü',
    'Beceri Kalıcılık Skoru',
    'Sınıf Düzeyi',
    'Kurum Politikası',
    'Dönem Öncesi GNO',
    'Dönem Sonrası GNO',
    'GNO Değişimi',
    'GNO Düşüş Şiddeti',
    'Beceri Kalıcılık Skoru Eksik Mi',
    'Prompt Yazma Becerisi Eksik Mi',
    'Dönem Öncesi GNO Eksik Mi',
    'Dönem Sonrası GNO Eksik Mi',
]

DOMAIN_CLIP_COLS = [
    'Haftalık AI Saati',
    'Geleneksel Çalışma Saati',
    'Dönem Öncesi GNO',
    'Dönem Sonrası GNO',
]

WEAK_MI_THRESHOLD = 0.0


class ProductionPreprocessor:
    """Ham girdi -> modele hazır sütunlar.

    Notlar:
    - KNN/Iterative imputation üretim tarafında pahalı olduğu için numeric
      sütunlarda median imputation kullanılır.
    - Eksiklik bilgisi ayrıca indicator feature olarak korunur.
    - Eşik tabanlı feature'ların eşikleri yalnızca train'de öğrenilir.
    """

    feature_names_: list[str] | None = None

    def __init__(
        self,
        weak_mi_threshold: float = WEAK_MI_THRESHOLD,
        use_compact_core: bool = True,
    ):
        self.weak_mi_threshold = weak_mi_threshold
        self.use_compact_core = use_compact_core

        self.numeric_imputer = SimpleImputer(strategy='median')
        self.scaler = RobustScaler()

        self.ordinal_imputer = SimpleImputer(strategy='most_frequent')
        self.ordinal_encoder = OrdinalEncoder(
            categories=ORDINAL_ORDERS,
            handle_unknown='use_encoded_value',
            unknown_value=-1,
        )

        self.ohe_imputer = SimpleImputer(strategy='most_frequent')
        self.ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        self.bool_imputer = SimpleImputer(strategy='most_frequent')

        self.ai_high_threshold_: float | None = None
        self.retention_low_threshold_: float | None = None
        self.full_feature_names_: list[str] | None = None
        self.selected_features_: list[str] | None = None
        self.dropped_weak_: list[str] | None = None
        self.mi_scores_: pd.Series | None = None

    def _fit_thresholds(self, X: pd.DataFrame) -> None:
        self.ai_high_threshold_ = float(X['Haftalık AI Saati'].median())
        self.retention_low_threshold_ = float(X['Beceri Kalıcılık Skoru'].quantile(0.35))

    def _threshold(self, attr_name: str, fallback: float) -> float:
        value = getattr(self, attr_name, None)
        return fallback if value is None or pd.isna(value) else float(value)

    def _add_engineered_features(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()

        ai_high_thr = self._threshold(
            'ai_high_threshold_',
            float(out['Haftalık AI Saati'].median()),
        )
        retention_low_thr = self._threshold(
            'retention_low_threshold_',
            float(out['Beceri Kalıcılık Skoru'].quantile(0.35)),
        )

        out['GNO Değişimi'] = out['Dönem Sonrası GNO'] - out['Dönem Öncesi GNO']
        out['GNO Ortalaması'] = (
            out['Dönem Sonrası GNO'] + out['Dönem Öncesi GNO']
        ) / 2
        out['GNO Düştü Mü'] = (out['GNO Değişimi'] < 0).astype(float)
        out['GNO Düşüş Şiddeti'] = (-out['GNO Değişimi']).clip(lower=0)

        out['AI / Çalışma Oranı'] = (
            out['Haftalık AI Saati'] / (out['Geleneksel Çalışma Saati'] + 1)
        )
        out['AI Çalışma Payı'] = (
            out['Haftalık AI Saati']
            / (out['Haftalık AI Saati'] + out['Geleneksel Çalışma Saati'] + 1)
        )
        out['Çalışma Dengesizliği'] = (
            out['Haftalık AI Saati'] - out['Geleneksel Çalışma Saati']
        )

        out['AI Bağımlılık Yükü'] = (
            out['Haftalık AI Saati'] * out['Algılanan AI Bağımlılığı']
        )
        out['Sınav Kaygısı × AI Saati'] = (
            out['Sınav Kaygı Düzeyi'] * out['Haftalık AI Saati']
        )
        out['AI × Kaygı × Bağımlılık'] = (
            out['Haftalık AI Saati']
            * out['Sınav Kaygı Düzeyi']
            * out['Algılanan AI Bağımlılığı']
        )

        out['Kalıcılık / AI Saati'] = (
            out['Beceri Kalıcılık Skoru'] / (out['Haftalık AI Saati'] + 1)
        )
        out['Kaygı / Kalıcılık'] = (
            out['Sınav Kaygı Düzeyi'] / (out['Beceri Kalıcılık Skoru'] + 1)
        )
        out['Bağımlılık / Kalıcılık'] = (
            out['Algılanan AI Bağımlılığı'] / (out['Beceri Kalıcılık Skoru'] + 1)
        )
        out['AI × Düşük Kalıcılık'] = (
            out['Haftalık AI Saati'] * (100 - out['Beceri Kalıcılık Skoru'])
        )

        out['Toplam Çalışma Yükü'] = (
            out['Haftalık AI Saati'] + out['Geleneksel Çalışma Saati']
        )

        out['Yüksek AI Kullanımı'] = (
            out['Haftalık AI Saati'] >= ai_high_thr
        ).astype(float)
        out['Yüksek Bağımlılık'] = (
            out['Algılanan AI Bağımlılığı'] >= 7
        ).astype(float)
        out['Yüksek Kaygı'] = (
            out['Sınav Kaygı Düzeyi'] >= 7
        ).astype(float)
        out['Düşük Kalıcılık'] = (
            out['Beceri Kalıcılık Skoru'] <= retention_low_thr
        ).astype(float)

        out['Yüksek AI + Yüksek Kaygı'] = (
            out['Yüksek AI Kullanımı'] * out['Yüksek Kaygı']
        )
        out['Yüksek AI + Düşük Kalıcılık'] = (
            out['Yüksek AI Kullanımı'] * out['Düşük Kalıcılık']
        )
        out['Yüksek AI + Yüksek Bağımlılık'] = (
            out['Yüksek AI Kullanımı'] * out['Yüksek Bağımlılık']
        )
        out['Risk Üçlüsü Bayrağı'] = (
            out['Yüksek AI Kullanımı']
            * out['Yüksek Kaygı']
            * out['Yüksek Bağımlılık']
        )
        out['Koruyucu Profil Bayrağı'] = (
            (out['Haftalık AI Saati'] < ai_high_thr)
            & (out['Algılanan AI Bağımlılığı'] <= 3)
            & (out['Beceri Kalıcılık Skoru'] > retention_low_thr)
        ).astype(float)

        out = out.replace([np.inf, -np.inf], np.nan)
        return out

    @staticmethod
    def _missing_indicators(X: pd.DataFrame) -> pd.DataFrame:
        data = {}
        for col in MISSING_INDICATOR_COLS:
            data[f'{col} Eksik Mi'] = X[col].isna().astype(float)
        return pd.DataFrame(data, index=X.index)

    @staticmethod
    def _coerce_bool(frame: pd.DataFrame) -> pd.DataFrame:
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

    def fit(self, X: pd.DataFrame, y_multiclass, y_binary_for_woe=None):
        self._fit_thresholds(X)
        X_fe = self._add_engineered_features(X)

        num_imputed = self.numeric_imputer.fit_transform(X_fe[ALL_NUMERIC])
        num_clipped = self._clip_domain(num_imputed)
        self.scaler.fit(num_clipped)

        ord_imputed = self.ordinal_imputer.fit_transform(X_fe[ORDINAL_COLS])
        self.ordinal_encoder.fit(ord_imputed)

        ohe_imputed = self.ohe_imputer.fit_transform(X_fe[OHE_COLS])
        self.ohe_encoder.fit(ohe_imputed)

        self.bool_imputer.fit(self._coerce_bool(X_fe[BOOL_COLS]))

        ohe_feature_names = self.ohe_encoder.get_feature_names_out(OHE_COLS).tolist()
        missing_feature_names = [f'{col} Eksik Mi' for col in MISSING_INDICATOR_COLS]

        self.full_feature_names_ = (
            ALL_NUMERIC
            + ORDINAL_COLS
            + ohe_feature_names
            + BOOL_COLS
            + missing_feature_names
        )

        full_df = self._transform_full_to_df(X)

        self.mi_scores_ = pd.Series(
            mutual_info_classif(full_df.values, y_multiclass, random_state=42),
            index=self.full_feature_names_,
        ).sort_values(ascending=False)

        if self.use_compact_core:
            missing_compact = [
                col for col in COMPACT_CORE_FEATURES
                if col not in self.full_feature_names_
            ]
            if missing_compact:
                raise ValueError(f'Compact core feature eksik: {missing_compact}')
            self.selected_features_ = COMPACT_CORE_FEATURES.copy()
        else:
            self.selected_features_ = self.full_feature_names_.copy()

        self.feature_names_ = self.selected_features_.copy()
        self.dropped_weak_ = []
        return self

    def _transform_full_to_df(self, X: pd.DataFrame) -> pd.DataFrame:
        X_fe = self._add_engineered_features(X)

        num_imputed = self.numeric_imputer.transform(X_fe[ALL_NUMERIC])
        num_clipped = self._clip_domain(num_imputed)
        num_scaled = self.scaler.transform(num_clipped)

        ord_imputed = self.ordinal_imputer.transform(X_fe[ORDINAL_COLS])
        ord_enc = self.ordinal_encoder.transform(ord_imputed)

        ohe_imputed = self.ohe_imputer.transform(X_fe[OHE_COLS])
        ohe_enc = self.ohe_encoder.transform(ohe_imputed)

        bool_imputed = self.bool_imputer.transform(self._coerce_bool(X_fe[BOOL_COLS]))
        bool_arr = pd.DataFrame(bool_imputed, columns=BOOL_COLS).astype(float).values

        missing_arr = self._missing_indicators(X).values

        full = np.hstack([
            num_scaled,
            ord_enc,
            ohe_enc,
            bool_arr,
            missing_arr,
        ])

        return pd.DataFrame(full, columns=self.full_feature_names_, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        full_df = self._transform_full_to_df(X)
        return full_df[self.selected_features_].copy()
