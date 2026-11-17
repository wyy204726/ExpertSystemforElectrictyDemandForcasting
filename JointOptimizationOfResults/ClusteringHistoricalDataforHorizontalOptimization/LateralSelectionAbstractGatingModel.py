# -*- coding: utf-8 -*-
"""
Clustering gating mechanism (extracted from "可视化展示附聚类机制.py";
all GUI/visualization code removed).

This module implements the gating selection mechanism of the ECHOES
expert system described in the manuscript:
  - Peak period (Jan, Feb, Jul, Aug, Sep, Nov, Dec): take the highest
    of the three expert predictions
  - Valley period (Mar, Apr, Oct): take the lowest
  - Normal period (May, Jun): take the prediction closest to the oil
    temperature of the K-means cluster center

Required precomputed files (under clustering_dir):
  - kmeans_model.pkl      K-means clustering model
  - standard_scaler.pkl   standardization scaler
  - cluster_centers.csv   oil temperature of each cluster center
                          (columns: cluster_id, center_temp)
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
import os
import warnings
import pickle
from datetime import datetime
import threading

warnings.filterwarnings('ignore')


# ---------- Period division ----------

def get_period_by_month(month):
    """
    Return the period type for a given month.
    Peak period:   Jan, Feb, Jul, Aug, Sep, Nov, Dec
    Valley period: Mar, Apr, Oct
    Normal period: May, Jun
    """
    peak_months = [1, 2, 7, 8, 9, 11, 12]
    valley_months = [3, 4, 10]

    if month in peak_months:
        return 'peak'
    elif month in valley_months:
        return 'valley'
    else:
        return 'normal'


def get_period_name(period_type):
    """Return the display name of the period."""
    if period_type == 'peak':
        return '高峰期'
    elif period_type == 'valley':
        return '低谷期'
    else:
        return '正常期'


# ---------- Clustering gating selector ----------

class ClusteringSelector:
    """Clustering-based gating selector."""

    def __init__(self, clustering_dir):
        self.clustering_dir = clustering_dir
        self.cluster_model = None
        self.scaler = None
        self.cluster_centers = None
        self.load_clustering_results()

    def load_clustering_results(self):
        """Load the precomputed clustering results."""
        try:
            model_path = os.path.join(self.clustering_dir, "kmeans_model.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"聚类模型文件不存在: {model_path}")

            with open(model_path, 'rb') as f:
                self.cluster_model = pickle.load(f)

            scaler_path = os.path.join(self.clustering_dir, "standard_scaler.pkl")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

            centers_path = os.path.join(self.clustering_dir, "cluster_centers.csv")
            centers_df = pd.read_csv(centers_path)
            self.cluster_centers = dict(zip(centers_df['cluster_id'], centers_df['center_temp']))

            print(f"聚类结果加载成功，共 {len(self.cluster_centers)} 个聚类")
            return True
        except Exception as e:
            print(f"加载聚类结果失败: {e}")
            return False

    def get_cluster_center(self, timestamp):
        """Return the oil temperature of the cluster center for a timestamp."""
        hour = timestamp.hour
        month = timestamp.month
        feature = np.array([[hour, month, 0]])
        feature_scaled = self.scaler.transform(feature)
        cluster_id = self.cluster_model.predict(feature_scaled)[0]
        cluster_center = self.cluster_centers[cluster_id]
        return cluster_id, cluster_center

    def select_best_predictions(self, predictions, dates):
        """
        Gating selection: determine the period type from the month and
        pick the final result from the three expert predictions.

        Args:
            predictions: [pred1, pred2, pred3], equal-length prediction
                arrays from the three models
            dates: pandas timestamp sequence aligned with the predictions
        Returns:
            final_predictions: gated final predictions (np.array)
            selection_details: per-point selection details (list[dict])
            period_labels: per-point period labels (list[str])
        """
        if len(predictions) != 3:
            raise ValueError("需要三个预测结果")

        final_predictions = []
        selection_details = []
        period_labels = []

        for i in range(len(predictions[0])):
            pred_values = [predictions[0][i], predictions[1][i], predictions[2][i]]
            timestamp = dates[i] if i < len(dates) else None

            if timestamp is None:
                period_type = 'normal'
            else:
                period_type = get_period_by_month(timestamp.month)

            period_labels.append(period_type)

            if period_type == 'peak':
                # Peak period: pick the highest temperature (overheating-warning oriented)
                selected_idx = np.argmax(pred_values)
                final_value = pred_values[selected_idx]
                selection_details.append({
                    'index': i, 'selected_model': selected_idx + 1,
                    'selected_value': final_value, 'reason': 'peak', 'period': period_type
                })
            elif period_type == 'valley':
                # Valley period: pick the lowest temperature
                selected_idx = np.argmin(pred_values)
                final_value = pred_values[selected_idx]
                selection_details.append({
                    'index': i, 'selected_model': selected_idx + 1,
                    'selected_value': final_value, 'reason': 'valley', 'period': period_type
                })
            else:
                if timestamp is None:
                    # Without a timestamp, fall back to the mean
                    final_value = np.mean(pred_values)
                    selection_details.append({
                        'index': i, 'selected_model': -1,
                        'selected_value': final_value, 'reason': 'normal_no_timestamp', 'period': period_type
                    })
                else:
                    # Normal period: pick the prediction closest to the cluster
                    # center; on a tie, pick the larger value
                    _, cluster_center = self.get_cluster_center(timestamp)
                    differences = [abs(pred - cluster_center) for pred in pred_values]
                    min_diff = min(differences)
                    candidates = [(pred, diff, idx + 1) for idx, (pred, diff) in
                                  enumerate(zip(pred_values, differences)) if diff == min_diff]
                    if len(candidates) == 1:
                        final_value = candidates[0][0]
                        selected_idx = candidates[0][2]
                    else:
                        final_value = max([c[0] for c in candidates])
                        selected_idx = [c[2] for c in candidates if c[0] == final_value][0]
                    selection_details.append({
                        'index': i, 'selected_model': selected_idx,
                        'selected_value': final_value, 'cluster_center': cluster_center,
                        'differences': differences, 'reason': 'normal_clustering', 'period': period_type
                    })
            final_predictions.append(final_value)

        return np.array(final_predictions), selection_details, period_labels


# ---------- Usage example ----------

if __name__ == "__main__":
    # Directory of the clustering results (adjust as needed)
    clustering_dir = r"D:\pythonProject\pythonProject\dataset\val\logs\clustering_results"

    selector = ClusteringSelector(clustering_dir)
    if selector.cluster_model is None:
        raise SystemExit("聚类结果未加载，请检查 clustering_dir 路径")

    # Example input: predictions of the three experts + aligned timestamps
    dates = pd.to_datetime([
        "2024-01-15 08:00",   # peak period
        "2024-04-15 14:00",   # valley period
        "2024-05-15 20:00",   # normal period
    ])
    pred1 = np.array([65.2, 38.1, 50.3])
    pred2 = np.array([64.8, 37.5, 49.8])
    pred3 = np.array([66.0, 38.8, 50.9])

    final_pred, details, labels = selector.select_best_predictions([pred1, pred2, pred3], dates)

    for d, lab, fp, det in zip(dates, labels, final_pred, details):
        print(f"{d}  {get_period_name(lab)}: 最终预测 = {fp:.3f}, "
              f"选中模型 {det['selected_model']}（规则: {det['reason']}）")
