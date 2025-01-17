from typing import Literal
import pandas as pd
import os
from typing import Iterable, List
import numpy as np


LEGACY_RESULTS_PATH = '/home/danielg/danigil/ModelXRay/results/legacy'

def get_legacy_supervised_results(
    chosen_zoo: str = "stl10", # mnist, cifar10, stl10, svhn
    msb: bool = False,

    ret_metric:Literal['f1', 'accuracy', 'precision', 'recall']='accuracy',
):
    df = pd.read_csv(os.path.join(LEGACY_RESULTS_PATH, f'results_supervised{"_msb" if msb else ""}.csv'))
    df_only_chosen_zoo = df[df["zoo"]==chosen_zoo]
    df_lsb_top_clf = df_only_chosen_zoo.loc[df_only_chosen_zoo.groupby('lsb')[ret_metric].idxmax()]
    df_lsb_top_clf.sort_values(by=['lsb'], inplace=True, ascending=[True])

    best_results = df_lsb_top_clf[ret_metric]
    return best_results



def weighted_metric(mal_acc: List[float], benign_acc: float) -> float:
    """
    benign_acc: accuracy of benign class
    mal_acc: list of `s` accuracies
    0.5 * benign_acc + 0.5 * (1/(1+2+...+s)) * (s*mal_acc_1 + (s-1)*mal_acc_2 + ... + 1*mal_acc_s)
    """
    s = len(mal_acc)
    if s == 0:
        mal_weighted_avg = 1.0
    
    else:
        weights = list(range(s, 0, -1))
        mal_weighted_avg = np.average(mal_acc, weights=weights)

    weighted_avg = np.average([benign_acc, mal_weighted_avg])

    return weighted_avg

def unweighted_metric(mal_acc: List[float], benign_acc: float) -> float:
    if len(mal_acc) == 0:
        mal_avg = 1.0
    else:
        mal_avg = np.average(mal_acc)
    unweighted_avg = np.average([benign_acc, mal_avg])

    return unweighted_avg

def calc_metrics(df, centroid=True, nn=True, weighted=True, unweighted=True):
    if len(df[df['lsb']==0]) == 0:
        benign_acc_centroid = 1
        benign_acc_nn = 1
    else:
        if centroid:
            benign_acc_centroid = df[df['lsb']==0]['test_acc_centroid'].iloc[0]
        if nn:
            benign_acc_nn = df[df['lsb']==0]['test_acc_nn'].iloc[0]

    df_mal = df[~(df['lsb'] == 0)].sort_values(by='lsb', ascending=True)

    metrics_dict = {}

    if centroid:
        mal_acc_centroid = df_mal['test_acc_centroid'].tolist()
        if unweighted:
            unweighted_metric_centroid = unweighted_metric(mal_acc_centroid, benign_acc_centroid)
            metrics_dict['unweighted_metric_centroid'] = unweighted_metric_centroid

        if weighted:
            weighted_metric_centroid = weighted_metric(mal_acc_centroid, benign_acc_centroid)
            metrics_dict['weighted_metric_centroid'] = weighted_metric_centroid
    if nn:
        mal_acc_nn = df_mal['test_acc_nn'].tolist()
        if unweighted:
            unweighted_metric_nn = unweighted_metric(mal_acc_nn, benign_acc_nn)
            metrics_dict['unweighted_metric_nn'] = unweighted_metric_nn

        if weighted:
            weighted_metric_nn = weighted_metric(mal_acc_nn, benign_acc_nn)
            metrics_dict['weighted_metric_nn'] = weighted_metric_nn

    return pd.Series(metrics_dict)