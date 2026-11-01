import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from mortgage_risk.data import synthetic_applications, validate_features, NUMERIC
from mortgage_risk.modeling import candidate_models, choose_threshold, evaluate


class PipelineTests(unittest.TestCase):
    def test_synthetic_reproducibility(self):
        a, b = synthetic_applications(500, 4), synthetic_applications(500, 4)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(set(a.default), {0, 1})
        self.assertTrue(a.isna().any().any())

    def test_schema_rejects_invalid_data(self):
        frame = synthetic_applications(100)
        with self.assertRaises(ValueError): validate_features(frame.drop(columns="credit_score"))
        frame.loc[0, "annual_income"] = np.inf
        with self.assertRaises(ValueError): validate_features(frame)

    def test_all_models_handle_missing_and_unknown_categories(self):
        frame = synthetic_applications(500, 7)
        x_train, x_test, y_train, _ = train_test_split(validate_features(frame), frame.default,
            test_size=.2, random_state=7, stratify=frame.default)
        x_test = x_test.copy()
        x_test.loc[:, "property_type"] = "unseen_property"
        for name, model in candidate_models(7).items():
            with self.subTest(model=name):
                model.fit(x_train, y_train)
                p = model.predict_proba(x_test)[:, 1]
                self.assertTrue(np.isfinite(p).all())
                self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_imputer_only_learns_training_data(self):
        frame = synthetic_applications(200, 9)
        train = validate_features(frame.iloc[:150])
        model = candidate_models()["logistic"].fit(train, frame.default.iloc[:150])
        statistics = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"].statistics_.copy()
        held_out = validate_features(frame.iloc[150:])
        held_out.loc[:, NUMERIC] = 1e8
        model.predict_proba(held_out)
        np.testing.assert_allclose(statistics, train[NUMERIC].median().to_numpy())
        np.testing.assert_array_equal(statistics, model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"].statistics_)

    def test_threshold_uses_stated_cost(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([.1, .4, .3, .9])
        threshold, cost = choose_threshold(y, p)
        self.assertLessEqual(threshold, .3)
        self.assertAlmostEqual(cost, .25)
        result = evaluate(y, p, threshold)
        self.assertEqual(result["confusion_matrix"][1][0], 0)

    def test_serialization_preserves_predictions(self):
        frame = synthetic_applications(200, 12)
        x = validate_features(frame)
        model = candidate_models()["logistic"].fit(x, frame.default)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
            np.testing.assert_array_equal(model.predict_proba(x), restored.predict_proba(x))


if __name__ == "__main__": unittest.main()
