from __future__ import annotations

from dataclasses import dataclass
from typing import Union, List, Dict, Set

import numpy as np
import pandas as pd
import sklearn.tree._tree as sklearn_tree_impl
import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text


@dataclass
class SimpleLeaf:
    value: float


@dataclass
class SimpleTree:
    input_index: int
    threshold: float
    left: Union[SimpleTree, SimpleLeaf]  # less or equal threshold
    right: Union[SimpleTree, SimpleLeaf]  # greater than threshold


def convert_tree_to_simple_tree_impl(
        tree: sklearn_tree_impl,
        node: int,
        features: Dict[int, int]
) -> Union[SimpleTree, SimpleLeaf]:
    if tree.feature[node] > 0:
        return SimpleTree(
            features[node],
            tree.threshold[node],
            convert_tree_to_simple_tree_impl(tree, tree.children_left[node], features),
            convert_tree_to_simple_tree_impl(tree, tree.children_right[node], features)
        )
    assert tree.n_outputs == 1
    assert len(tree.value[node][0]) == 1
    return SimpleLeaf(tree.value[node][0][0])


def convert_tree_to_simple_tree(tree: sklearn_tree_impl) -> SimpleTree:
    result = convert_tree_to_simple_tree_impl(tree, 0, dict(enumerate(tree.feature)))
    assert isinstance(result, SimpleTree)
    return result


def simple_tree_features_used(simple_tree: Union[SimpleTree, SimpleLeaf]) -> Set[int]:
    if isinstance(simple_tree, SimpleLeaf):
        return set()
    return {simple_tree.input_index}.union(
        simple_tree_features_used(simple_tree.left).union(
            simple_tree_features_used(simple_tree.right)))


def show_simple_tree_as_python_code(simple_tree: Union[SimpleTree, SimpleLeaf], depth: int = 0) -> str:
    indent = "    " * depth
    if isinstance(simple_tree, SimpleLeaf):
        return f"{indent}return {simple_tree.value}"
    threshold = simple_tree.threshold
    left = show_simple_tree_as_python_code(simple_tree.left, depth + 1)
    right = show_simple_tree_as_python_code(simple_tree.right, depth + 1)
    return f"{indent}if x_{simple_tree.input_index} <= {threshold}:\n{left}\n{indent}else:\n{right}"


def make_leaf_layer(value: float) -> tf.keras.layers.Layer:
    return tf.keras.layers.Dense(
        1,
        "linear",
        kernel_initializer=tf.keras.initializers.Constant(value),
        bias_initializer=tf.keras.initializers.Constant(0.0)
    )


def make_switch_layer_left(switched_input: KerasTensor, threshold: float, edginess: float) -> KerasTensor:
    """return fuzzy 1 if x < threshold"""
    return tf.keras.layers.Dense(
        1, "sigmoid",
        kernel_initializer=tf.keras.initializers.Constant(-edginess),
        bias_initializer=tf.keras.initializers.Constant(threshold * edginess)
    )(switched_input)


def make_switch_layer_right(switched_input: KerasTensor, threshold: float, edginess: float) -> KerasTensor:
    """return fuzzy 1 if x > threshold"""
    return tf.keras.layers.Dense(
        1, "sigmoid",
        kernel_initializer=tf.keras.initializers.Constant(edginess),
        bias_initializer=tf.keras.initializers.Constant(-threshold * edginess)
    )(switched_input)


def simple_tree_as_neural_network_impl(
        node: Union[SimpleTree, SimpleLeaf],
        inputs: List[tf.keras.layers.Input],
        on_off_signal: KerasTensor,
        edginess: float
) -> List[KerasTensor]:
    if isinstance(node, SimpleLeaf):
        return [make_leaf_layer(node.value)(on_off_signal)]
    left_on_off = tf.keras.layers.Multiply()([
        on_off_signal,
        make_switch_layer_left(inputs[node.input_index], node.threshold, edginess)])
    right_on_off = tf.keras.layers.Multiply()([
        on_off_signal,
        make_switch_layer_right(inputs[node.input_index], node.threshold, edginess)])
    left = simple_tree_as_neural_network_impl(node.left, inputs, left_on_off, edginess)
    right = simple_tree_as_neural_network_impl(node.right, inputs, right_on_off, edginess)
    return left + right


def simple_tree_as_neural_network(
        simple_tree: SimpleTree,
        num_input_features: int,
        edginess: float = 10.0
) -> tf.keras.Model:
    always_on = tf.constant(1.0, shape=(1,))
    feature_inputs = [tf.keras.layers.Input(shape=(1,)) for _ in range(num_input_features)]
    inputs = feature_inputs
    tree_outputs = simple_tree_as_neural_network_impl(simple_tree, inputs, always_on, edginess)
    outputs = [tf.keras.layers.Add()(tree_outputs)]
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def tree_input_to_nn_input(data):
    return list(np.swapaxes(data, 0, 1))


def treebominate():
    pass