import unittest
import shutil

import numpy as np
import pandas as pd
from click.testing import CliRunner

from rsschool_mlintro2022q1_capstone_project.train import train
from rsschool_mlintro2022q1_capstone_project.predict import predict

from .vars_test import (
    TEST_DATA_DIR,
    TEST_TRAIN_PATH,
    TEST_TRAIN_DATA,
    TEST_PRED_PATH,
    TEST_PRED_DATA
)


class TestTrain(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if not TEST_DATA_DIR.exists():
            TEST_DATA_DIR.mkdir(parents=True)
        with open(TEST_TRAIN_PATH, "w") as file:
            file.write(TEST_TRAIN_DATA)
        with open(TEST_PRED_PATH, "w") as file:
            file.write(TEST_PRED_DATA)
        cls.runner = CliRunner()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(TEST_DATA_DIR)

    def test_train_wrong_input(self) -> None:
        commands = (
            ["--k-folds", 0],
            ["--scaler", "not_existed_scaler"],
            ["--k-best", -1],
        )
        for i_test, command in enumerate(commands):
            with self.subTest(i_test=i_test, command=command):
                result = self.runner.invoke(
                    train, command
                )
                self.assertEqual(
                    result.exit_code, 2, f"Error not raised on {command}"
                )
                self.assertIn(
                    f"Invalid value for '{command[0]}'", result.output
                )

    def test_train_wrong_input_double(self) -> None:
        commands = (
            ["--dataset-path", "test_data/not_existed.csv"],
            ["--model", "not_existed_model"],
        )
        for i_test, command in enumerate(commands):
            with self.subTest(i_test=i_test, command=command):
                result = self.runner.invoke(
                    train, command
                )
                short = command[0][2]
                self.assertEqual(
                    result.exit_code, 2, f"Error not raised on {command}"
                )
                self.assertIn(
                    f"Invalid value for '-{short}' / '{command[0]}'",
                    result.output
                )

    def test_a_train(self):
        result = self.runner.invoke(
            train, [
                "-d", TEST_TRAIN_PATH,
                "-s", TEST_DATA_DIR / "test_model.joblib",
                "--k-folds", 2
            ]
        )
        self.assertEqual(result.exit_code, 0, result.exc_info[1])
        self.assertTrue((TEST_DATA_DIR / "test_model.joblib").exists())

    def test_b_predict(self):
        result = self.runner.invoke(
            predict, [
                "-d", TEST_PRED_PATH,
                "-m", TEST_DATA_DIR / "test_model.joblib",
                "-s", TEST_DATA_DIR / "test_sub.csv"
            ]
        )
        self.assertEqual(result.exit_code, 0, result.exc_info[1])
        self.assertTrue((TEST_DATA_DIR / "test_sub.csv").exists())

        true = np.column_stack((
            range(1, 11), [1, 1, 0, 0, 1, 1, 0, 0, 0, 0]
        ))
        pred = pd.read_csv(TEST_DATA_DIR / "test_sub.csv").to_numpy()
        self.assertTrue(np.all(true == pred))
