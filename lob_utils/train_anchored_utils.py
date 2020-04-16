from lob_utils.lob_loader import get_wf_lob_loaders
from lob_utils.lob_model_utils import epoch_trainer as lob_epoch_trainer
from lob_utils.lob_model_utils import evaluator as lob_evaluator
import numpy as np


def train_evaluate_anchored(model, epoch_trainer=lob_epoch_trainer, evaluator=lob_evaluator,
                            horizon=0, window=5, batch_size=128, train_epochs=20, verbose=True,
                            use_resampling=True, learning_rate=0.0001, splits=range(9)):
    """
    Trains and evaluates a model for using an anchored walk-forward setup
    :param model: model to train
    :param epoch_trainer: function to use for training the model (please refer to lob.model_utils.epoch_trainer() )
    :param evaluator: function to use for evaluating the model (please refer to lob.model_utils.epoch_trainer() )
    :param horizon: the prediction horizon for the evaluation (0, 5 or 10)
    :param window: the window to use
    :param batch_size: batch size to be used
    :param train_epochs: number of epochs for training the model
    :param intermediate_evaluation: if set to True, the model will be evaluated after each epoch
    :param verbose:
    :return:
    """

    results = []

    for i in splits:
        print("Evaluating for split: ", i)
        train_loader, test_loader = get_wf_lob_loaders(window=window, horizon=horizon, split=i, batch_size=batch_size,
                                                       class_resample=use_resampling)
        current_model = model()
        current_model.cuda()
        for epoch in range(train_epochs):
            loss = epoch_trainer(model=current_model, loader=train_loader, use_class_weights=(not use_resampling),
                                 lr=learning_rate)
            if verbose:
                print("Epoch ", epoch, "loss: ", loss)

        test_results = evaluator(model=current_model, loader=test_loader)
        print(test_results)
        results.append(test_results)

    return results


def get_average_metrics(results):
    precision, recall, f1 = [], [], []
    kappa = []
    acc = []
    for x in results:
        acc.append(x['accuracy'])
        precision.append(x['precision_avg'])
        recall.append(x['recall_avg'])
        f1.append(x['f1_avg'])
        kappa.append(x['kappa'])

    return acc, precision, recall, f1, kappa
