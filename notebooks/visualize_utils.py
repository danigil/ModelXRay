from typing import Literal
import pandas as pd
import os

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

