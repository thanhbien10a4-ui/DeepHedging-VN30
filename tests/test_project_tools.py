import importlib.util
import json
import tempfile
import unittest
import zipfile
import hashlib
from pathlib import Path

spec = importlib.util.spec_from_file_location('project_tools', Path(__file__).resolve().parents[1] / 'scripts/project.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EvidenceTests(unittest.TestCase):
    def test_missing_blocks_release(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(module.audit(root)['status'], 'INCOMPLETE')
            with self.assertRaises(RuntimeError):
                module.package(root, False, False)

    def test_invalid_json_and_failed_sanity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name, text in [('lstm_variance_baseline_report.json', '{broken'),
                               ('lstm_cvar_turnover_report.json', '{"all_sanity_checks_pass": false}')]:
                p = root / 'results/metrics' / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text)
            self.assertEqual(len(module.audit(root)['errors']), 2)

    def test_diagnostic_bundle_excludes_secret_and_has_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / '.env').write_text('SECRET=not-for-bundle')
            target = module.package(root, False, True)
            with zipfile.ZipFile(target) as z:
                self.assertNotIn('.env', z.namelist())
                manifest = json.loads(z.read('MANIFEST.json'))
                self.assertEqual(manifest['audit_status'], 'INCOMPLETE')
                for item in manifest['files']:
                    self.assertEqual(hashlib.sha256(z.read(item['path'])).hexdigest(), item['sha256'])

    def test_nonfinite_and_empty_csv(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = root / 'results/metrics/environment_snapshot.json'
            p.parent.mkdir(parents=True)
            p.write_text('{"metric": NaN}')
            p = root / 'results/tables/baseline_validation_metrics.csv'
            p.parent.mkdir(parents=True)
            p.write_text('strategy,value\n')
            self.assertEqual(len(module.audit(root)['errors']), 2)


if __name__ == '__main__':
    unittest.main()
