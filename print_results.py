from os.path import join
import pickle
import numpy as np
from lob_utils.train_anchored_utils import  get_average_metrics

def print_line(results):
    metrics = get_average_metrics(results)
    acc, precision, recall, f1, kappa = metrics

    print("$ %3.2f \\pm %3.2f$ & $ %3.2f \\pm %3.2f$ & $ %3.2f \\pm %3.2f$ & $ %3.2f \\pm %3.2f $ & $ %5.4f \\pm %5.4f $"
          % (100*np.mean(acc), 100*np.std(acc),
             100*np.mean(precision), 100*np.std(precision),
             100 * np.mean(recall), 100 * np.std(recall),
             100 * np.mean(f1), 100 * np.std(f1),
             np.mean(kappa), np.std(kappa)))

def print_results(results_path):
    with open(join("results", results_path), 'rb') as f:
        [metrics_1, metrics_2, metrics_3] = pickle.load(f)
        [results1, results2, results3] = pickle.load(f)

    print("--------")
    print(results_path)
    print_line(results1)
    # print_line(results2)
    # print_line(results3)



print_results("final_cnn.pickle")
print_results("final_gru.pickle")
# print_results("final_lstm.pickle")
print_results("final_bof.pickle")


