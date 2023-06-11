![logo](treebomination.jpg)

[![CI](https://github.com/Dobiasd/treebomination/workflows/ci/badge.svg)](https://github.com/Dobiasd/treebomination/actions)
[![(License MIT 1.0)](https://img.shields.io/badge/license-MIT%201.0-blue.svg)][license]

[license]: LICENSE

# treebomination

Treebomination is a way to convert a `sklearn.tree.DecisionTreeRegressor` into a (roughly) equivalent `tf.keras.Model`.

## When is this helpful?

- You irrationally dislike decision trees and feel neural networks are much cooler.
- You want to prove a point about neural networks.
- You think that converting the tree to a NN and then fine-tuning it might decrease the value of your less metric on your test set.
- You have a well-working decision tree but want to only use TensorFlow or [frugally-deep](https://github.com/Dobiasd/frugally-deep) in production.
- You want to back up the claims of your marketing department about your team using "AI".

## When is not useful?
- You care about the performance of your predictions.
- You care about the precision of your results.
- You care about the size of your final application.

So, **I highly recommend not actually using this for anything serious**.

## Usage

```bash
from sklearn.tree import DecisionTreeRegressor
from treebomination import treebominate

my_decision_tree_regressor = DecisionTreeRegressor()
# ... training ...
model = treebominate(my_decision_tree_regressor)
```

## Origin story

From some unbridled thoughts:
- A Decision tree is a fancy way of having nested `if` statements.
- A simple logistic regression on a one-dimensional input acts like a fuzzy threshold (or an `if` statement).
- A neuron in an artificial neural network acts can act as a single logistic regression node.

The following idea arose: There should be a morphism from binary decision trees to neural networks, it should™️ be possible to emulate every decision tree with a neural network, i.e., derive the network architecture from the tree and initialize the weights and biases such that the output of the network is similar to the output of the tree.