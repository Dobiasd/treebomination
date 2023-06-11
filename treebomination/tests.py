import unittest

from ._conversion import treebominate, tree_input_to_nn_input


class TestArgsCalls(unittest.TestCase):

    def test_foo_function_positional(self) -> None:
        treebominate()
        tree_input_to_nn_input()
