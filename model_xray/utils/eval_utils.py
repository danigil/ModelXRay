import numpy as np
from model_xray.config_classes import ModelRepos, ClassificationMetric, ClassificationMetricConfig

def _top_k_accuracy(y_true, y_pred, k=1):
    argsorted_y = np.argsort(y_pred)[:,-k:]

    assert y_true.ndim == 2, f"_top_k_accuracy expects y_true to be one-hot-encoded, got shape {y_true.shape}"
    assert y_pred.ndim == 2, f"_top_k_accuracy expects y_pred to be one-hot-encoded, got shape {y_pred.shape}"

    # print(y_true.shape)
    # print(y_pred.shape)

    return np.any(argsorted_y.T == y_true.argmax(axis=1), axis=0).mean()

def calc_metric(y_true, y_pred, metric_cfg: ClassificationMetricConfig) -> float:
    metric_type = metric_cfg.metric_type
    
    if metric_type == ClassificationMetric.TopKCategoricalAccuracy:
        return _top_k_accuracy(y_true, y_pred, k=metric_cfg.classification_metric_config.k)
    else:
        raise NotImplementedError(f'calc_metric | metric_type {metric_type} not implemented')