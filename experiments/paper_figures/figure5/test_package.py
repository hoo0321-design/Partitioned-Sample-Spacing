"""Small integrity/recomputation/relocation checks; never retrain a model."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import plot


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pss-figure5-test-")
        self.root = Path(self.temp.name)
        self.package = self.root / "relocated_figure"
        self.package.mkdir()
        shutil.copy2(plot.HERE / "plot.py", self.package / "plot.py")
        shutil.copytree(plot.HERE / "data", self.package / "data")

    def tearDown(self):
        self.temp.cleanup()

    def test_relocated_cli_verifies_and_plots(self):
        def run(*options):
            completed = subprocess.run([sys.executable, str(self.package / "plot.py"), *options],
                                       cwd=self.root, check=True, capture_output=True, text=True)
            return json.loads(completed.stdout)
        before = set(self.package.rglob("*"))
        result = run("--verify-only")
        self.assertEqual(result["prediction_metrics_rechecked"], 240)
        self.assertEqual(result["summarized_points"], 80)
        self.assertFalse(result["full_energy_csv_checked"])
        self.assertEqual(before, set(self.package.rglob("*")))
        output = self.root / "fresh_output"
        result = run("--output-dir", str(output))
        self.assertEqual(result["status"], "PASS")
        for suffix in ["pdf", "png", "svg"]:
            self.assertGreater((output / f"energy_four_methods.{suffix}").stat().st_size, 1000)
        for line in (output / "energy_four_methods.svg").read_text().splitlines():
            self.assertEqual(line, line.rstrip())
        for name in ["metrics.csv", "summary.csv"]:
            self.assertNotIn(b"\r", (self.package / "data" / name).read_bytes())

    def test_changed_file_fails_hash(self):
        with (self.package / "data/metrics.csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch: metrics.csv"):
            plot.load_verified(self.package / "data")

    def test_shallow_standalone_location(self):
        # A Linux /tmp/figure5 directory has fewer than three indexed parents.
        with patch.object(plot, "HERE", Path("/tmp/figure5")):
            _, checks = plot.load_verified(self.package / "data")
        self.assertEqual(checks["status"], "PASS")
        self.assertFalse(checks["full_energy_csv_checked"])

    def test_altered_prediction_fails_recalculation_even_with_updated_hash(self):
        path = self.package / "data/predictions.npz"
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays["predictions"][0, 0, 0, 0] ^= 1
        np.savez_compressed(path, **arrays)
        manifest_path = self.package / "data/provenance.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifact_sha256"]["predictions.npz"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Saved accuracy differs from predictions"):
            plot.load_verified(self.package / "data")


if __name__ == "__main__":
    unittest.main()
