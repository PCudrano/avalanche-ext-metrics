################################################################################
# Copyright (c) 2023 Paolo Cudrano.                                            #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 03-07-2023                                                             #
# Author(s): Paolo Cudrano                                                     #
# E-mail: paolo.cudrano@polimi.it                                              #
#                                                                              #
# Inspired by code under:                                                      #
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 30-12-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from typing import List, Union, Dict, Any

import torch
from torch import Tensor
from avalanche.evaluation import Metric, PluginMetric, GenericPluginMetric
from avalanche.evaluation.metrics.mean import Mean
from avalanche.evaluation.metric_utils import phase_and_task
from collections import defaultdict

import torchmetrics


class TorchMetric(Metric[float]):
    """Torchmetrics metric. This is a standalone metric.

    The update method computes the accuracy incrementally
    by keeping a running average of the <prediction, target> pairs
    of Tensors provided over time.

    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.

    Each time `result` is called, this metric emits the average accuracy
    across all predictions made since the last `reset`.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an accuracy value of 0.
    """

    def __init__(self, torchmetric_class, torchmetrics_name=None, **torchmetrics_kwargs):
        """Creates an instance of the standalone Torchmetric metric.

        By default this metric in its initial state will return an accuracy
        value of 0. The metric can be updated by using the `update` method
        while the running accuracy can be retrieved using the `result` method.
        """
        self._metric = torchmetric_class(**torchmetrics_kwargs)

        self.name = torchmetrics_name if torchmetrics_name is not None else torchmetric_class.__name__

        """The mean utility that will be used to store the running accuracy."""

    @torch.no_grad()
    def update(
        self,
        predicted_y: Tensor,
        true_y: Tensor,
    ) -> None:
        """Update the running metric given the true and predicted labels.

        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.

        :return: None.
        """
        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y and predicted_y tensors")

        # # Check if logits or labels
        # if len(predicted_y.shape) > 1:
        #     # Logits -> transform to labels
        #     predicted_y = torch.max(predicted_y, 1)[1]
        #
        # if len(true_y.shape) > 1:
        #     # Logits -> transform to labels
        #     true_y = torch.max(true_y, 1)[1]

        self._metric.update(predicted_y, true_y)

    def result(self) -> float:
        """Retrieves the running metric.

        Calling this method will not change the internal state of the metric.

        :return: The current running accuracy, which is a float value
            between 0 and 1.
        """
        return self._metric.compute()

    def reset(self) -> None:
        """Resets the metric.

        :return: None.
        """
        self._metric.reset()

    def to(self, device) -> None:
        self._metric.to(device)

class TorchMetricPluginMetric(GenericPluginMetric[float]):
    """
    Base class for all torchmetrics plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode, torchmetrics_class, torchmetrics_name=None, **torchmetrics_kwargs): # , split_by_task=False):
        """Creates the Accuracy plugin

        :param reset_at:
        :param emit_at:
        :param mode:
        :param split_by_task: whether to compute task-aware accuracy or not.
        """
        # self.split_by_task = split_by_task
        # if self.split_by_task:
        #     self._accuracy = TaskAwareAccuracy()
        # else:
        self._metric = TorchMetric(torchmetrics_class, torchmetrics_name, **torchmetrics_kwargs)
        super(TorchMetricPluginMetric, self).__init__(
            self._metric, reset_at=reset_at, emit_at=emit_at, mode=mode
        )

    def reset(self, strategy=None) -> None:
        self._metric.reset()

    def result(self, strategy=None) -> float:
        res = self._metric.result()
        if isinstance(res, torch.Tensor):
            res = res.cpu()
        return float(res)

    def update(self, strategy):
        if isinstance(self._metric, TorchMetric):
            self._metric.update(strategy.mb_output, strategy.mb_y)
        # elif isinstance(self._accuracy, TaskAwareAccuracy):
        #     self._accuracy.update(
        #         strategy.mb_output, strategy.mb_y, strategy.mb_task_id
        #     )
        else:
            assert False, "should never get here."

    def before_training(self, strategy, **kwargs):
        self._metric.to(strategy.device)

class MinibatchTorchMetric(TorchMetricPluginMetric):
    """
    The minibatch plugin torchmetric metric.
    This metric only works at training time.

    This metric computes the average torchmetric over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochAccuracy` instead.
    """

    def __init__(self, torchmetrics_class, torchmetrics_name=None, **torchmetrics_kwargs):
        """
        Creates an instance of the MinibatchAccuracy metric.
        """
        super(MinibatchTorchMetric, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train",
            torchmetrics_class=torchmetrics_class, torchmetrics_name=torchmetrics_name, **torchmetrics_kwargs
        )

    def __str__(self):
        # return "Top1_Acc_MB"
        return f'{self._metric.name}_MB'


class EpochTorchMetric(TorchMetricPluginMetric):
    """
    The average torchmetric over a single training epoch.
    This plugin metric only works at training time.

    The torchmetric will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self, torchmetrics_class, torchmetrics_name=None, **torchmetrics_kwargs):
        """
        Creates an instance of the EpochTorchMetric metric.
        """

        super(EpochTorchMetric, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train",
            torchmetrics_class=torchmetrics_class, torchmetrics_name=torchmetrics_name, **torchmetrics_kwargs
        )

    def __str__(self):
        # return "Top1_Acc_Epoch"
        return f'{self._metric.name}_Epoch'


class RunningEpochTorchMetric(TorchMetricPluginMetric):
    """
    The average torchmetric across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the accuracy averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self, torchmetrics_class, torchmetrics_name=None, **torchmetrics_kwargs):
        """
        Creates an instance of the RunningEpochTorchMetric metric.
        """

        super(RunningEpochTorchMetric, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train",
            torchmetrics_class=torchmetrics_class, torchmetrics_name=torchmetrics_name, **torchmetrics_kwargs\
        )

    def __str__(self):
        # return "Top1_RunningAcc_Epoch"
        return f'{self._metric.name}Running_Epoch'


class ExperienceTorchMetric(TorchMetricPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average accuracy over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self, torchmetrics_class, torchmetrics_name=None, **torchmetrics_kwargs):
        """
        Creates an instance of ExperienceTorchMetric metric
        """
        super(ExperienceTorchMetric, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval",
            torchmetrics_class=torchmetrics_class, torchmetrics_name=torchmetrics_name, **torchmetrics_kwargs
        )

    def __str__(self):
        # return "Top1_Acc_Exp"
        return f'{self._metric.name}_Exp'


class StreamTorchMetric(TorchMetricPluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average torchmetric over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self, torchmetrics_class, torchmetrics_name=None, **torchmetrics_kwargs):
        """
        Creates an instance of StreamAccuracy metric
        """
        super(StreamTorchMetric, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval",
            torchmetrics_class=torchmetrics_class, torchmetrics_name=torchmetrics_name, **torchmetrics_kwargs
        )

    def __str__(self):
        # return "Top1_Acc_Stream"
        return f'{self._metric.name}_Stream'


class TrainedExperienceTorchMetric(TorchMetricPluginMetric):
    """
    At the end of each experience, this plugin metric reports the average
    torchmetric for only the experiences that the model has been trained on so far.

    This metric only works at eval time.
    """

    def __init__(self, torchmetrics_class, torchmetrics_name=None, **torchmetrics_kwargs):
        """
        Creates an instance of TrainedExperienceTorchMetric metric by first
        constructing TorchMetricPluginMetric
        """
        super(TrainedExperienceTorchMetric, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval",
            torchmetrics_class=torchmetrics_class, torchmetrics_name=torchmetrics_name, **torchmetrics_kwargs
        )
        self._current_experience = 0

    def after_training_exp(self, strategy) -> None:
        self._current_experience = strategy.experience.current_experience
        # Reset average after learning from a new experience
        TorchMetricPluginMetric.reset(self, strategy)
        return TorchMetricPluginMetric.after_training_exp(self, strategy)

    def update(self, strategy):
        """
        Only update the accuracy with results from experiences that have been
        trained on
        """
        if strategy.experience.current_experience <= self._current_experience:
            TorchMetricPluginMetric.update(self, strategy)

    def __str__(self):
        # return "Accuracy_On_Trained_Experiences"
        return f'{self._metric.name}_On_Trained_Experiences'


def torchmetrics_metrics(
    torchmetrics_class,
    torchmetrics_name=None,
    minibatch=False,
    epoch=False,
    epoch_running=False,
    experience=False,
    stream=False,
    trained_experience=False,
    **torchmetrics_kwargs,
) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param minibatch: If True, will return a metric able to log
        the minibatch accuracy at training time.
    :param epoch: If True, will return a metric able to log
        the epoch accuracy at training time.
    :param epoch_running: If True, will return a metric able to log
        the running epoch accuracy at training time.
    :param experience: If True, will return a metric able to log
        the accuracy on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the accuracy averaged over the entire evaluation stream of experiences.
    :param trained_experience: If True, will return a metric able to log
        the average evaluation accuracy only for experiences that the
        model has been trained on

    :return: A list of plugin metrics.
    """

    metrics = []
    if minibatch:
        metrics.append(MinibatchTorchMetric(torchmetrics_class, torchmetrics_name, **torchmetrics_kwargs))

    if epoch:
        metrics.append(EpochTorchMetric(torchmetrics_class, torchmetrics_name, **torchmetrics_kwargs))

    if epoch_running:
        metrics.append(RunningEpochTorchMetric(torchmetrics_class, torchmetrics_name, **torchmetrics_kwargs))

    if experience:
        metrics.append(ExperienceTorchMetric(torchmetrics_class, torchmetrics_name, **torchmetrics_kwargs))

    if stream:
        metrics.append(StreamTorchMetric(torchmetrics_class, torchmetrics_name, **torchmetrics_kwargs))

    if trained_experience:
        metrics.append(TrainedExperienceTorchMetric(torchmetrics_class, torchmetrics_name, **torchmetrics_kwargs))

    return metrics


__all__ = [
    "TorchMetric",
    # "TaskAwareAccuracy",
    "MinibatchTorchMetric",
    "EpochTorchMetric",
    "RunningEpochTorchMetric",
    "ExperienceTorchMetric",
    "StreamTorchMetric",
    "TrainedExperienceTorchMetric",
    "torchmetrics_metrics",
]