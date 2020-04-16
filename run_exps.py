from models.bof_models import ConvolutionalTemporalCorrelationBoFAdaptivePyramid
from lob_utils.train_anchored_utils import train_evaluate_anchored, get_average_metrics
from time import time
import pickle
from models.nn_models import  GRU_NN, LSTM_NN, CNN_NN
from os.path import join


def run_experiments(model, output_path, train_epochs=20, window=10):

    a = time()
    results1 = train_evaluate_anchored(model, window=window, train_epochs=train_epochs, horizon=0, splits=range(9))
    b = time()

    print("----------")
    print("Elapsed time = ", b - a)
    metrics_1 = get_average_metrics(results1)

    print("----------")

    with open(join("results", output_path), 'wb') as f:
        pickle.dump([metrics_1, metrics_1, metrics_1], f)
        pickle.dump([results1, results1, results1], f)


# Experiments!

model = lambda: ConvolutionalTemporalCorrelationBoFAdaptivePyramid(window=15, split_horizon=5, use_scaling=True)
run_experiments(model, 'final_bof.pickle', window=15)

model = lambda: GRU_NN()
run_experiments(model, 'final_gru.pickle', window=15)

model = lambda: LSTM_NN()
run_experiments(model, 'final_lstm.pickle', window=15)

model = lambda: CNN_NN()
run_experiments(model, 'final_cnn.pickle', window=15)



