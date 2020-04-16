import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score


def epoch_trainer(model, loader, lr=0.001, optimizer=optim.Adam, use_class_weights=False):
    model.train()

    model_optimizer = optimizer(model.parameters(), lr=lr)
    if use_class_weights:
        weight = np.float32([1, 0.05, 1])
        weight = Variable(torch.from_numpy(weight).cuda())
        criterion = CrossEntropyLoss()
    else:
        criterion = CrossEntropyLoss()

    train_loss, counter = 0, 0

    for (inputs, targets) in loader:
        # Reset gradients
        model_optimizer.zero_grad()

        # Get the data
        inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
        targets = torch.squeeze(targets)

        # Feed forward the network and update
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        model_optimizer.step()

        # Calculate statistics
        train_loss += loss.item()
        counter += inputs.size(0)

    loss = (loss / counter).cpu().data.numpy()
    return loss


def evaluator(model, loader):
    model.eval()
    true_labels = []
    predicted_labels = []

    for (inputs, targets) in tqdm(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        predicted_labels.append(predicted.cpu().numpy())
        true_labels.append(targets.cpu().data.numpy())

    true_labels = np.squeeze(np.concatenate(true_labels))
    predicted_labels = np.squeeze(np.concatenate(predicted_labels))

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average=None)
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(true_labels, predicted_labels,
                                                                           average='macro')
    kappa = cohen_kappa_score(true_labels, predicted_labels)

    metrics = {}
    metrics['accuracy'] = np.sum(true_labels == predicted_labels) / len(true_labels)

    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1

    metrics['precision_avg'] = precision_avg
    metrics['recall_avg'] = recall_avg
    metrics['f1_avg'] = f1_avg

    metrics['kappa'] = kappa

    return metrics
