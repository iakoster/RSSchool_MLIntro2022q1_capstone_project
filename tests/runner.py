import unittest
from pathlib import Path


def start_tests():
    _tests_loader = unittest.TestLoader()
    all_tests_suite = unittest.TestSuite()
    all_tests_suite.addTests(
        _tests_loader.discover(
            str(Path(".\\tests")), top_level_dir="."
        )
    )
    result = unittest.TestResult()
    all_tests_suite.run(result, debug=True)
