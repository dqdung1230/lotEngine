import numpy as np

def evaluate_predictions(pred_main_indices, true_main_vec, pred_special, true_special):
    actual_main_nums = np.where(true_main_vec == 1)[0] + 1
    hits_main = len(set(pred_main_indices).intersection(set(actual_main_nums)))
    hit_special = 1 if pred_special == (true_special + 1) else 0
    return hits_main, hit_special
