################################################################################
# Copyright (c) 2023 Paolo Cudrano.                                            #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 03-07-2023                                                             #
# Author(s): Paolo Cudrano                                                     #
# E-mail: paolo.cudrano@polimi.it                                              #
#                                                                              #
# Adapted from code under:                                                     #
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 30-12-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

"""
This is a simple example on how to use the Evaluation Plugin.
"""

import torch
import argparse
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomCrop

from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.classic import PermutedMNIST, RotatedMNIST, SplitMNIST
from avalanche.benchmarks.datasets.dataset_utils import default_dataset_location
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    labels_repartition_metrics,
    loss_metrics,
    cpu_usage_metrics,
    timing_metrics,
    gpu_usage_metrics,
    ram_usage_metrics,
    disk_usage_metrics,
    MAC_metrics,
    bwt_metrics,
    forward_transfer_metrics,
    class_accuracy_metrics,
    amca_metrics,
)
from avalanche.models import SimpleMLP
from avalanche.logging import (
    InteractiveLogger,
    TextLogger,
    CSVLogger,
    TensorboardLogger,
)
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive

from evaluation.metrics.torchmetrics_metrics import *
from evaluation.metrics.sklearn_metrics import *
import torchmetrics
import sklearn.metrics as skmetrics


def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )
    # ---------
    print(device)

    n_classes = 10
    benchmark = PermutedMNIST(n_experiences=5, seed=1)

    # MODEL CREATION
    model = SimpleMLP(num_classes=n_classes)

    # DEFINE THE EVALUATION PLUGIN AND LOGGER
    # The evaluation plugin manages the metrics computation.
    # It takes as argument a list of metrics and a list of loggers.
    # The evaluation plugin calls the loggers to serialize the metrics
    # and save them in persistent memory or print them in the standard output.

    # log to text file
    text_logger = TextLogger(open("./log.txt", "a"))

    # print to stdout
    interactive_logger = InteractiveLogger()

    csv_logger = CSVLogger()

    tb_logger = TensorboardLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        loss_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        torchmetrics_metrics(torchmetrics.F1Score, task="multiclass", num_classes=n_classes, threshold=0.5,
                             minibatch=True,
                             epoch=True,
                             epoch_running=True,
                             experience=True,
                             stream=True
                             ),
        torchmetrics_metrics(torchmetrics.AUROC, task="multiclass", num_classes=n_classes, thresholds=200,
                             minibatch=True,
                             epoch=True,
                             epoch_running=True,
                             experience=True,
                             stream=True
                             ),
        torchmetrics_metrics(torchmetrics.Precision, task="multiclass", num_classes=n_classes, threshold=0.5,
                             minibatch=True,
                             epoch=True,
                             epoch_running=True,
                             experience=True,
                             stream=True
                             ),
        torchmetrics_metrics(torchmetrics.Recall, task="multiclass", num_classes=n_classes, threshold=0.5,
                             minibatch=True,
                             epoch=True,
                             epoch_running=True,
                             experience=True,
                             stream=True
                             ),
        sklearn_metrics(skmetrics.roc_auc_score, average='macro', multi_class='ovr',
                        use_logits=True, running_average=False, regression=False,
                        minibatch=True,
                        epoch=True,
                        epoch_running=True,
                        experience=True,
                        stream=True
                        ),
        sklearn_metrics(skmetrics.f1_score, average='micro',
                        use_logits=False, running_average=False,
                        minibatch=True,
                        epoch=True,
                        epoch_running=True,
                        experience=True,
                        stream=True
                        ),
        sklearn_metrics(skmetrics.precision_score, average='micro',
                        use_logits=False, running_average=False, regression=False,
                        minibatch=True,
                        epoch=True,
                        epoch_running=True,
                        experience=True,
                        stream=True
                        ),
        sklearn_metrics(skmetrics.recall_score, average='micro',
                        use_logits=False, running_average=False, regression=False,
                        minibatch=True,
                        epoch=True,
                        epoch_running=True,
                        experience=True,
                        stream=True
                        ),
        sklearn_metrics(skmetrics.accuracy_score,
                        use_logits=False, running_average=False, regression=False,
                        minibatch=True,
                        epoch=True,
                        epoch_running=True,
                        experience=True,
                        stream=True
                        ),
        loggers=[interactive_logger, text_logger, csv_logger, tb_logger],
        collect_all=True,
    )  # collect all metrics (set to True by default)

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = Naive(
        model,
        SGD(model.parameters(), lr=0.01, momentum=0.9),
        CrossEntropyLoss(),
        train_mb_size=512,
        train_epochs=1,
        eval_mb_size=512,
        device=device,
        evaluator=eval_plugin,
        eval_every=1,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    for i, experience in enumerate(benchmark.train_stream):
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        # train returns a dictionary containing last recorded value
        # for each metric.
        res = cl_strategy.train(experience)  # , eval_streams=[benchmark.test_stream])
        print("Training completed")

        print("Computing accuracy on the whole test set")
        # test returns a dictionary with the last metric collected during
        # evaluation on that stream
        results.append(cl_strategy.eval(benchmark.test_stream))

    print(f"Test metrics:\n{results}")

    # Dict with all the metric curves,
    # only available when `collect_all` is True.
    # Each entry is a (x, metric value) tuple.
    # You can use this dictionary to manipulate the
    # metrics without avalanche.
    all_metrics = cl_strategy.evaluator.get_all_metrics()
    print(f"Stored metrics: {list(all_metrics.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args)