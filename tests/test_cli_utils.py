import argparse
import unittest

from cli_utils import str2bool


class StrToBoolTests(unittest.TestCase):
    def test_true_values(self):
        for value in ('true', 'TRUE', '1', 'yes', 'y', 'on'):
            with self.subTest(value=value):
                self.assertTrue(str2bool(value))

    def test_false_values(self):
        for value in ('false', 'FALSE', '0', 'no', 'n', 'off'):
            with self.subTest(value=value):
                self.assertFalse(str2bool(value))

    def test_bool_values_are_unchanged(self):
        self.assertTrue(str2bool(True))
        self.assertFalse(str2bool(False))

    def test_invalid_value_is_rejected(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            str2bool('sometimes')


if __name__ == '__main__':
    unittest.main()
