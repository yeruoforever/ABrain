import unittest
import os

import torchio
from torchio import Subject
import nibabel as nib

from ...evaluation.base import Analyzer, ComposeAnalyzer


class TestAnalyzer(unittest.TestCase):
    def setUp(self) -> None:
        self.database = "tinyset"
        self.img = os.path.join(self.database, "1", "orig", "FLAIR.nii.gz")
        self.seg_1 = os.path.join(self.database, "1", "segm.nii.gz")
        self.seg_2 = os.path.join(self.database, "2", "segm.nii.gz")
        self.subject = Subject(
            name="sub 1",
            img=torchio.ScalarImage(self.img)
        )

    def test_analyer(self):
        with self.assertRaises(NotImplementedError):
            a = Analyzer("origin analyzer")
            a(self.subject)

    def test_analyzer_subclass(self):
        class TestAnalyzer(Analyzer):
            def __init__(self) -> None:
                super().__init__("Name Test")

            def analyze(self, subject: Subject):
                return subject["name"]
        ta = TestAnalyzer()
        ans = ta(self.subject)
        self.assertIsInstance(ans, dict)
        self.assertEqual(len(ans), 1+1)
        self.assertTrue(True)

    def test_compose(self):
        class TestAnalyzer(Analyzer):
            def __init__(self, name) -> None:
                super().__init__(name)

            def analyze(self, subject: Subject):
                return subject["name"]

        a = ComposeAnalyzer(
            TestAnalyzer("T1"),
            TestAnalyzer("T2"),
            TestAnalyzer("T3"),
        )
        ans = a(self.subject)
        self.assertEqual(len(ans), 1+3)
        self.assertIn("T1", ans)
        self.assertIn("T2", ans)
        self.assertIn("T3", ans)
