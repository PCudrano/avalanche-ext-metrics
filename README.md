# avalanche-ext-metrics

Wrappers to allow usage of torchmetrics and sklearn within [Avalanche](https://avalanche.continualai.org/). 

## Motivation
Avalanche provides an easy-to-use environment for computing and logging metrics.<br>
For example, you can log the accuracy on several time-scales with just this snippet:
```
eval_plugin = EvaluationPlugin(
    accuracy_metrics(
        minibatch=True,
        epoch=True,
        experience=True,
        stream=True,
        epoch_running=True,
    )
)
```
This tells Avalanche to keep track and log the accuracy at every minibatch, but also at every epoch, at every experience, for the entire stream, and to keep a running average throughout each epoch[^1].<br>
This is very convenient and easy to use, instead of defining manually each single metric and keeping track manually of their accumulation.

Problem: Avalanche provides only a limited set of [these metrics](https://avalanche-api.continualai.org/en/v0.3.1/evaluation.html) (and they are all mainly for classification).

On the other hand, [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/) is a well-known library specialized in computing all sorts of metrics efficiently and in a user-friendly way.
[Scikit-learn Metrics](https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics) is also very well-known and provides a wide range of metrics used in many benchmarks. 

This repo extends the classic set of metrics in Avalanche, allowing to use any metric from torchmetrics and sklearn.metrics using the same Avalanche syntax.

[^1]: https://avalanche-api.continualai.org/en/v0.3.1/generated/avalanche.evaluation.metrics.accuracy_metrics.html#avalanche.evaluation.metrics.accuracy_metrics

## Usage


### Torchmetrics
```
from evaluation.metrics.torchmetrics_metrics import *

eval_plugin = EvaluationPlugin(
    
    # ... other metrics ...
    
    torchmetrics_metrics( # This wrapper
        
        torchmetrics_class, # class of the torchmetrics metric
        
        **torchmetrics_class_kwargs, # any argument required by the torchmetrics class
        
        # usual avalanche helper arguments to determine when to compute/log this metric
        minibatch=True,
        epoch=True,
        experience=True
        stream=True,
        epoch_running=True
        )
)
```

### Sklearn (experimental)
```
from evaluation.metrics.sklearn_metrics import *

eval_plugin = EvaluationPlugin(
    
    # ... other metrics ...
    
    sklearn( # This wrapper
        torchmetrics_class, # class of the torchmetrics metric
        
        **torchmetrics_class_kwargs, # any argument required by the torchmetrics class
         
        regression=False, # whether it is a regression metric (if false, assumer classification)
        
        use_logits=True, # only if classification metric: whether to pass to sklearn the logits or to extract first a class label through softmax+argmax
        
        running_average=False, # whether the metric can be computed with a running average; if false, all values are stored until a call to compute() is internally made (this can increase the memory cost)
        
        # usual avalanche helper arguments to determine when to compute/log this metric
        minibatch=True,
        epoch=True,
        experience=True
        stream=True,
        epoch_running=True
        )
)
```
