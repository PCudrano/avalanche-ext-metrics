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


class SklearnMetric(Metric[float]):
    """sklearn metric. This is a standalone metric.

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

    def __init__(self, sklearn_fcn, metric_name=None, use_logits=True, regression=False, running_average=False, **sklearn_kwargs):
        """Creates an instance of the standalone sklearn metric.

        By default this metric in its initial state will return an accuracy
        value of 0. The metric can be updated by using the `update` method
        while the running accuracy can be retrieved using the `result` method.
        """
        self._metric_fcn = sklearn_fcn
        self._metric_kwargs = sklearn_kwargs
        self.name = metric_name if metric_name is not None else sklearn_fcn.__name__
        self.use_logits = use_logits
        self.running_average = running_average
        self.regression = regression

        if self.running_average:
            self._mean = Mean()
        else:
            self._target = torch.tensor([])
            self._pred = torch.tensor([])

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

        if not self.regression: # if classification metric
            # # Check if logits or labels
            if len(predicted_y.shape) > 1:
                # Logits -> transform to logits
                predicted_y = torch.softmax(predicted_y, -1)
                if predicted_y.shape[1] == 2: # binary
                    predicted_y = predicted_y[:,1] # take only logit for class 1

            if not self.use_logits:
                # predicted_y = torch.round(predicted_y)
                predicted_y = torch.max(predicted_y, 1)[1]

            if len(true_y.shape) > 1:
                # Logits -> transform to labels
                true_y = torch.max(true_y, 1)[1]

        predicted_y = predicted_y.cpu()
        true_y = true_y.cpu()

        if self.running_average:
            total_patterns = len(true_y)
            computed = self._metric_fcn(true_y, predicted_y, **self._metric_kwargs)
            self._mean.update(computed, total_patterns)
        else:
            self._pred = torch.cat((self._pred, predicted_y), dim=0)
            self._target = torch.cat((self._target, true_y), dim=0)

    def result(self) -> float:
        if self.running_average:
            return self._mean.result()
        else:
            #try:
            return self._metric_fcn(self._target, self._pred, **self._metric_kwargs)
            # except Exception as e:
            #     return 0.0 # But error

    def reset(self) -> None:
        """Resets the metric.

        :return: None.
        """
        if self.running_average:
            self._mean.reset()
        else:
            self._target = torch.tensor([])
            self._pred = torch.tensor([])


class SklearnMetricPluginMetric(GenericPluginMetric[float]):
    """
    Base class for all torchmetrics plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode, sklearn_fcn, metric_name=None, **kwargs): # , split_by_task=False):
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
        self._metric = SklearnMetric(sklearn_fcn, metric_name, **kwargs)
        super(SklearnMetricPluginMetric, self).__init__(
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
        if isinstance(self._metric, SklearnMetric):
            self._metric.update(strategy.mb_output, strategy.mb_y)
        # elif isinstance(self._accuracy, TaskAwareAccuracy):
        #     self._accuracy.update(
        #         strategy.mb_output, strategy.mb_y, strategy.mb_task_id
        #     )
        else:
            assert False, "should never get here."

class MinibatchSklearnMetric(SklearnMetricPluginMetric):
    """
    The minibatch plugin sklearn metric.
    This metric only works at training time.

    This metric computes the average sklearn over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochAccuracy` instead.
    """

    def __init__(self, sklearn_fcn, metric_name=None, **kwargs):
        """
        Creates an instance of the MinibatchAccuracy metric.
        """
        super(MinibatchSklearnMetric, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train",
            sklearn_fcn=sklearn_fcn, metric_name=metric_name, **kwargs
        )

    def __str__(self):
        # return "Top1_Acc_MB"
        return f'{self._metric.name}_MB'


class EpochSklearnMetric(SklearnMetricPluginMetric):
    """
    The average sklearn over a single training epoch.
    This plugin metric only works at training time.

    The sklearn will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self, sklearn_fcn, metric_name=None, **kwargs):
        """
        Creates an instance of the EpochSklearnMetric metric.
        """

        super(EpochSklearnMetric, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train",
            sklearn_fcn=sklearn_fcn, metric_name=metric_name, **kwargs
        )

    def __str__(self):
        # return "Top1_Acc_Epoch"
        return f'{self._metric.name}_Epoch'


class RunningEpochSklearnMetric(SklearnMetricPluginMetric):
    """
    The average sklearn across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the accuracy averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self, sklearn_fcn, metric_name=None, **kwargs):
        """
        Creates an instance of the RunningEpochSklearnMetric metric.
        """

        super(RunningEpochSklearnMetric, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train",
            sklearn_fcn=sklearn_fcn, metric_name=metric_name, **kwargs
        )

    def __str__(self):
        # return "Top1_RunningAcc_Epoch"
        return f'{self._metric.name}Running_Epoch'


class ExperienceSklearnMetric(SklearnMetricPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average accuracy over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self, sklearn_fcn, metric_name=None, **kwargs):
        """
        Creates an instance of ExperienceSklearnMetric metric
        """
        super(ExperienceSklearnMetric, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval",
            sklearn_fcn=sklearn_fcn, metric_name=metric_name, **kwargs
        )

    def __str__(self):
        # return "Top1_Acc_Exp"
        return f'{self._metric.name}_Exp'


class StreamSklearnMetric(SklearnMetricPluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average sklearn over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self, sklearn_fcn, metric_name=None, **kwargs):
        """
        Creates an instance of StreamAccuracy metric
        """
        super(StreamSklearnMetric, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval",
            sklearn_fcn=sklearn_fcn, metric_name=metric_name, **kwargs
        )

    def __str__(self):
        # return "Top1_Acc_Stream"
        return f'{self._metric.name}_Stream'


class TrainedExperienceSklearnMetric(SklearnMetricPluginMetric):
    """
    At the end of each experience, this plugin metric reports the average
    sklearn for only the experiences that the model has been trained on so far.

    This metric only works at eval time.
    """

    def __init__(self, sklearn_fcn, metric_name=None, **kwargs):
        """
        Creates an instance of TrainedExperienceSklearnMetric metric by first
        constructing SklearnMetricPluginMetric
        """
        super(TrainedExperienceSklearnMetric, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval",
            sklearn_fcn=sklearn_fcn, metric_name=metric_name, **kwargs
        )
        self._current_experience = 0

    def after_training_exp(self, strategy) -> None:
        self._current_experience = strategy.experience.current_experience
        # Reset average after learning from a new experience
        SklearnMetricPluginMetric.reset(self, strategy)
        return SklearnMetricPluginMetric.after_training_exp(self, strategy)

    def update(self, strategy):
        """
        Only update the accuracy with results from experiences that have been
        trained on
        """
        if strategy.experience.current_experience <= self._current_experience:
            SklearnMetricPluginMetric.update(self, strategy)

    def __str__(self):
        # return "Accuracy_On_Trained_Experiences"
        return f'{self._metric.name}_On_Trained_Experiences'


def sklearn_metrics(
    sklearn_fcn,
    metric_name=None,
    minibatch=False,
    epoch=False,
    epoch_running=False,
    experience=False,
    stream=False,
    trained_experience=False,
    **kwargs
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
        metrics.append(MinibatchSklearnMetric(sklearn_fcn, metric_name, **kwargs))

    if epoch:
        metrics.append(EpochSklearnMetric(sklearn_fcn, metric_name, **kwargs))

    if epoch_running:
        metrics.append(RunningEpochSklearnMetric(sklearn_fcn, metric_name, **kwargs))

    if experience:
        metrics.append(ExperienceSklearnMetric(sklearn_fcn, metric_name, **kwargs))

    if stream:
        metrics.append(StreamSklearnMetric(sklearn_fcn, metric_name, **kwargs))

    if trained_experience:
        metrics.append(TrainedExperienceSklearnMetric(sklearn_fcn, metric_name, **kwargs))

    return metrics


__all__ = [
    "SklearnMetric",
    # "TaskAwareAccuracy",
    "MinibatchSklearnMetric",
    "EpochSklearnMetric",
    "RunningEpochSklearnMetric",
    "ExperienceSklearnMetric",
    "StreamSklearnMetric",
    "TrainedExperienceSklearnMetric",
    "sklearn_metrics",
]