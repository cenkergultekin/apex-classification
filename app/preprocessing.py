# -*- coding: utf-8 -*-
"""Production-safe preprocessor — Lean Core (Binary + WoE + StandardScaler).

Ham formdan gelen DataFrame doğrudan ``ProductionPreprocessor.transform`` ile
modele hazır hale gelir. Öğrenilen tüm eşikler/imputer/encoder/scaler yalnızca
``fit`` sırasında verilen train verisinden öğrenilir.

Phase 3 kararları (binary geçiş + Lean Core refactor sonrası):

- Hedef: Binary (Yüksek=1, Düşük=0). 'Orta' satırları modelleme dışı.
- GPA bloğu (Pre/Post + türevleri) **drop** — ablation'da Yüksek risk recall'unu
  düşürdüğü ve canlı tahmin senaryolarında kırılgan olduğu için.
- Nominal kategoriler (Okunan Bölüm, Birincil Kullanım Amacı) → WoE Encoder
  (binary log-odds, tek sütun).
- Ordinal kategoriler (Sınıf Düzeyi, Prompt Yazma Becerisi, Kurum Politikası)
  → OrdinalEncoder (sıra korunur).
- Boolean (Ücretli Abonelik) → int 0/1.
- Tüm sayısal+ordinal+WoE → StandardScaler. Aykırı oranı %0.5 altında olduğu
  için RobustScaler şart değil.
- 7 redundant türev feature **drop** — yüksek-MI değişkenlerden ratio/product
  türetildikleri için kolinearite üretiyorlardı. Tree-based modeller bu
  etkileşimleri zaten kendileri öğrenir.
- Sadece ``Toplam Çalışma Yükü`` korundu — sum, kapasite/yük kavramı taşır.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from category_encoders import WOEEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


# ── Sütun tanımları (Lean Core)

BASE_NUMERIC = [
    'Haftalık AI Saati',
    'Algılanan AI Bağımlılığı',
    'Sınav Kaygı Düzeyi',
    'Geleneksel Çalışma Saati',
    'Beceri Kalıcılık Skoru',
    'Araç Çeşitliliği',
]

ENGINEERED_NUMERIC = [
    'Toplam Çalışma Yükü',
]

ALL_NUMERIC = BASE_NUMERIC + ENGINEERED_NUMERIC

ORDINAL_COLS = ['Sınıf Düzeyi', 'Prompt Yazma Becerisi', 'Kurum Politikası']
ORDINAL_ORDERS = [
    ['1. Sınıf', '2. Sınıf', '3. Sınıf', '4. Sınıf', 'Yüksek Lisans'],
    ['Başlangıç', 'Orta', 'İleri'],
    ['Kesin Yasak', 'Kaynak Belirterek İzinli', 'Aktif Olarak Teşvik'],
]

WOE_COLS = ['Okunan Bölüm', 'Birincil Kullanım Amacı']
BOOL_COLS = ['Ücretli Abonelik']

MISSING_INDICATOR_COLS = ['Beceri Kalıcılık Skoru', 'Prompt Yazma Becerisi']

DOMAIN_CLIP_COLS = ['Haftalık AI Saati', 'Geleneksel Çalışma Saati']


class ProductionPreprocessor:
    """Ham girdi → modele hazır Lean Core sütunlar.

    - WoE nominal encoding (binary log-odds)
    - OrdinalEncoder hiyerarşik sütunlar
    - StandardScaler (sayısal + ordinal + WoE)
    - Toplam Çalışma Yükü engineered feature
    - Missing indicators (Beceri Kalıcılık + Prompt Yazma Becerisi)
    - Sadece train'de fit; test sızıntısı yok.

    Lean Core (Phase 3 refactor) — her zaman 15 sabit feature üretir.
    """

    feature_names_: list[str] | None = None

    def __init__(self):
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()

        self.ordinal_imputer = SimpleImputer(strategy='most_frequent')
        self.ordinal_encoder = OrdinalEncoder(
            categories=ORDINAL_ORDERS,
            handle_unknown='use_encoded_value',
            unknown_value=-1,
        )

        self.woe_imputer = SimpleImputer(strategy='most_frequent')
        self.woe_encoder = WOEEncoder(cols=WOE_COLS)

        self.bool_imputer = SimpleImputer(strategy='most_frequent')

        self.feature_names_: list[str] | None = None
        self.selected_features_: list[str] | None = None

    # ── Helper transforms

    def _add_engineered_features(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out['Toplam Çalışma Yükü'] = (
            out['Haftalık AI Saati'].fillna(out['Haftalık AI Saati'].median())
            + out['Geleneksel Çalışma Saati'].fillna(out['Geleneksel Çalışma Saati'].median())
        )
        out = out.replace([np.inf, -np.inf], np.nan)
        return out

    @staticmethod
    def _missing_indicators(X: pd.DataFrame) -> pd.DataFrame:
        data = {}
        for col in MISSING_INDICATOR_COLS:
            indicator_name = f'{col} Eksik Mi'
            data[indicator_name] = X[col].isna().astype(float) if col in X.columns else 0.0
        return pd.DataFrame(data, index=X.index)

    @staticmethod
    def _coerce_bool(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col in out.columns:
            s = out[col]
            if s.dtype == 'bool':
                out[col] = s.astype(float)
            elif str(s.dtype) in ('boolean', 'Int64'):
                out[col] = s.astype('Float64').astype(float)
            else:
                out[col] = pd.to_numeric(s, errors='coerce')
        return out

    def _clip_domain(self, num_df: pd.DataFrame) -> pd.DataFrame:
        out = num_df.copy()
        for col in DOMAIN_CLIP_COLS:
            if col in out.columns:
                out[col] = out[col].clip(lower=0)
        return out

    # ── Fit / Transform

    def fit(self, X: pd.DataFrame, y, y_binary_for_woe=None):
        """Train'de fit eder.

        Parameters
        ----------
        X : pd.DataFrame
            Ham train özellikleri.
        y : array-like
            Binary hedef (0=Düşük, 1=Yüksek) veya çok sınıflı (binary'ye dönüştürülür).
        y_binary_for_woe : array-like, optional
            Backward-compat; verilirse y yerine WoE fit için kullanılır.
        """
        # y'yi binary'ye normalize et
        y_binary = y_binary_for_woe if y_binary_for_woe is not None else y
        y_binary = np.asarray(y_binary)
        if y_binary.dtype.kind in ('U', 'O'):
            y_binary = (pd.Series(y_binary) == 'Yüksek').astype(int).values
        else:
            # Eğer multi-class (0/1/2) gelirse Yüksek=2 varsayımıyla binary'ye çevir
            unique_vals = set(np.unique(y_binary))
            if unique_vals - {0, 1}:
                # Multi-class — en yüksek değer pozitif kabul edilir
                y_binary = (y_binary == max(unique_vals)).astype(int)
            else:
                y_binary = y_binary.astype(int)

        X_fe = self._add_engineered_features(X)

        # Numeric imputer + scaler (sadece numeric block için)
        num_imputed = self.numeric_imputer.fit_transform(X_fe[ALL_NUMERIC])
        num_imp_df = pd.DataFrame(num_imputed, columns=ALL_NUMERIC, index=X.index)
        num_clipped = self._clip_domain(num_imp_df)
        # Scaler tüm Lean Core sayısal sinyaller için fit edilir (sayısal + ordinal + WoE)
        # Önce sayısal kısmını öğret, sonra transform sırasında genişletilmiş matriste fit-transform yapacağız.
        # Burada sadece sayısal scaler için fit:
        self.scaler_numeric_only_ = StandardScaler().fit(num_clipped.values)

        # Ordinal imputer + encoder
        ord_imputed = self.ordinal_imputer.fit_transform(X_fe[ORDINAL_COLS])
        self.ordinal_encoder.fit(ord_imputed)

        # WoE imputer + encoder (y_binary gerektirir)
        woe_imputed = self.woe_imputer.fit_transform(X_fe[WOE_COLS])
        woe_imp_df = pd.DataFrame(woe_imputed, columns=WOE_COLS, index=X.index)
        self.woe_encoder.fit(woe_imp_df, pd.Series(y_binary, index=X.index))

        # Bool imputer
        self.bool_imputer.fit(self._coerce_bool(X_fe[BOOL_COLS]))

        # Final feature list (sıralı)
        missing_feature_names = [f'{col} Eksik Mi' for col in MISSING_INDICATOR_COLS]
        feature_order = (
            ALL_NUMERIC
            + ORDINAL_COLS
            + WOE_COLS
            + BOOL_COLS
            + missing_feature_names
        )

        # StandardScaler'ı tüm scaled bloka uygula (binary missing + bool hariç)
        full_df_pre_scale = self._transform_to_df_pre_scale(X)
        scale_cols = ALL_NUMERIC + ORDINAL_COLS + WOE_COLS
        self.scaler = StandardScaler()
        self.scaler.fit(full_df_pre_scale[scale_cols].values)

        self.feature_names_ = feature_order
        self.selected_features_ = feature_order.copy()
        return self

    def _transform_to_df_pre_scale(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scaling öncesi tüm encoding'leri uygular."""
        X_fe = self._add_engineered_features(X)

        num_imputed = self.numeric_imputer.transform(X_fe[ALL_NUMERIC])
        num_df = pd.DataFrame(num_imputed, columns=ALL_NUMERIC, index=X.index)
        num_df = self._clip_domain(num_df)

        ord_imputed = self.ordinal_imputer.transform(X_fe[ORDINAL_COLS])
        ord_enc = self.ordinal_encoder.transform(ord_imputed)
        ord_df = pd.DataFrame(ord_enc, columns=ORDINAL_COLS, index=X.index)

        woe_imputed = self.woe_imputer.transform(X_fe[WOE_COLS])
        woe_imp_df = pd.DataFrame(woe_imputed, columns=WOE_COLS, index=X.index)
        woe_enc_df = self.woe_encoder.transform(woe_imp_df)
        woe_enc_df.index = X.index

        bool_imputed = self.bool_imputer.transform(self._coerce_bool(X_fe[BOOL_COLS]))
        bool_df = pd.DataFrame(bool_imputed, columns=BOOL_COLS, index=X.index).astype(float)

        missing_df = self._missing_indicators(X)

        out = pd.concat([num_df, ord_df, woe_enc_df, bool_df, missing_df], axis=1)
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pre_scale = self._transform_to_df_pre_scale(X)
        scale_cols = ALL_NUMERIC + ORDINAL_COLS + WOE_COLS
        scaled = self.scaler.transform(pre_scale[scale_cols].values)
        out = pre_scale.copy()
        out[scale_cols] = scaled
        # Kolon sırası garantisi
        return out[self.feature_names_].copy()
