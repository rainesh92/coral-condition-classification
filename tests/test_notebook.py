"""Exercise pure notebook functions without downloading data or loading PyTorch."""
import ast
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / 'coral_multilabel_classification.ipynb').read_text())


def source(index):
    return ''.join(NOTEBOOK['cells'][index]['source'])


def load_function(index, name):
    tree = ast.parse(source(index))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(np=np, pd=pd, Image=Image, SEED=42, tqdm=lambda x, **kw: x,
                     f1_score=f1_score)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<notebook>', 'exec'), namespace)
    return namespace[name]


class NotebookTests(unittest.TestCase):
    def test_all_code_cells_compile(self):
        for i, cell in enumerate(NOTEBOOK['cells']):
            if cell['cell_type'] == 'code':
                with self.subTest(cell=i):
                    compile(source(i), f'cell-{i}', 'exec')

    def test_pixel_variance_includes_differences_between_images(self):
        stats = load_function(16, 'compute_dataset_stats')
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for value in (0, 255):
                path = Path(directory) / f'{value}.png'
                Image.new('RGB', (8, 8), (value,) * 3).save(path)
                paths.append(str(path))
            mean, std = stats(paths)
            np.testing.assert_allclose(mean, [0.5] * 3)
            np.testing.assert_allclose(std, [0.5] * 3)

    def test_unreadable_images_fail_clearly(self):
        stats = load_function(16, 'compute_dataset_stats')
        with self.assertRaisesRegex(ValueError, 'Geen leesbare'):
            stats(['/missing/image.png'])

    def test_constant_channels_fail_clearly(self):
        stats = load_function(16, 'compute_dataset_stats')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'constant.png'
            Image.new('RGB', (8, 8), (42,) * 3).save(path)
            with self.assertRaisesRegex(ValueError, 'geen variatie'):
                stats([str(path)])

    def test_tuning_keeps_exact_default_boundary(self):
        tune = load_function(29, 'tune_thresholds_per_label')
        y = np.array([[0], [1], [0]])
        p = np.array([[0.49], [0.5], [0.51]], dtype=np.float32)
        for grid in (None, [0.1, 0.9]):
            thresholds, _ = tune(y, p, ['healthy'], grid=grid)
            self.assertEqual(float(thresholds[0]), 0.5)
            self.assertEqual(f1_score(y, p >= thresholds), f1_score(y, p >= 0.5))

    def test_invalid_threshold_rejected(self):
        tune = load_function(29, 'tune_thresholds_per_label')
        with self.assertRaises(ValueError):
            tune(np.array([[1]]), np.array([[0.5]]), ['healthy'], grid=[np.nan])

    def test_missing_overfit_check_is_unverified(self):
        tree = ast.parse(source(68))
        checks = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'checks' for t in n.targets))
        expression = next(v for k, v in zip(checks.value.keys, checks.value.values)
                          if k.value.startswith('4.'))
        for gap, expected in [(float('nan'), None), (float('inf'), None), (0.1, True), (0.3, False)]:
            self.assertIs(eval(compile(ast.Expression(expression), '<check>', 'eval'),
                               {'math': math, 'overfit_gap': gap}), expected)


if __name__ == '__main__':
    unittest.main()
