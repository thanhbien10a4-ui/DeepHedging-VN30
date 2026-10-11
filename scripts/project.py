"""Run frozen evaluation, audit evidence and package a research handoff.

Standard-library only. Never automatically train or tune a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = [
    'lstm_variance_baseline_report.json', 'lstm_cvar_turnover_report.json',
    'real_finetune_lstm_report.json', 'final_oos_real_test_report.json',
    'cost_aware_oos_frozen_report.json', 'market_regime_oos_frozen_report.json',
    'transaction_cost_ablation_frozen_report.json', 'graceful_degradation_frozen_report.json',
    'frozen_oos_performance_reporting.json', 'full_market_simulator_frozen_report.json',
    'environment_snapshot.json',
]
REQUIRED = [*(f'results/metrics/{n}' for n in REPORTS),
    'data/processed/baseline_validation_report.json',
    'data/processed/deep_hedging/sequence_metadata.json',
    'data/processed/calibration/frozen_simulator_config.json',
    'data/synthetic/frozen/metadata.json',
    'results/models/lstm_cvar_turnover_best.pt',
    'results/models/lstm_cvar_turnover_real_finetuned_best.pt',
    'results/tables/baseline_validation_metrics.csv',
    'results/tables/lstm_variance_training_history.csv',
    'results/tables/lstm_cvar_turnover_training_history.csv',
    'results/tables/real_finetune_training_history.csv',
    'results/tables/full_market_simulator_frozen_summary.csv',
    'results/tables/market_regime_oos_frozen.csv',
    'results/tables/transaction_cost_ablation_frozen.csv',
    'results/tables/graceful_degradation_final_summary.csv',
    'results/figures/frozen_oos_mean_cumulative_performance.png',
    'results/figures/frozen_oos_drawdown_distribution.png',
    'results/figures/lstm_cvar_turnover_training_curve.png',
    'results/figures/real_finetune_cvar_training_curve.png',
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def finite(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def audit(root: Path) -> dict:
    missing, errors, warnings, files = [], [], [], []
    for name in REQUIRED:
        p = root / name
        if not p.is_file() or p.stat().st_size == 0:
            missing.append(name)
            continue
        files.append({'path': name, 'bytes': p.stat().st_size, 'sha256': digest(p)})
        try:
            if p.suffix == '.json':
                data = json.loads(p.read_text(encoding='utf-8-sig'))
                if not isinstance(data, dict) or not finite(data):
                    errors.append(f'{name}: expected a finite JSON object')
                if data.get('all_sanity_checks_pass') is False:
                    errors.append(f'{name}: sanity checks failed')
                if data.get('fine_tuning_improved_over_epoch0') is False:
                    warnings.append('Fine-tuning retained epoch 0; do not claim improvement.')
                if data.get('model_retuned') is True:
                    errors.append(f'{name}: model_retuned must be false for frozen evaluation')
            elif p.suffix == '.csv':
                with p.open(encoding='utf-8-sig', newline='') as f:
                    reader = csv.DictReader(f)
                    if not reader.fieldnames or not any(reader):
                        errors.append(f'{name}: empty CSV')
        except (ValueError, OSError, csv.Error) as exc:
            errors.append(f'{name}: {exc}')
    warnings.append('Artifact checks do not prove convergence, provenance or statistical superiority; human review is required.')
    return {'status': 'INCOMPLETE' if missing or errors else 'ARTIFACT_CHECKS_PASS_REVIEW_REQUIRED',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'missing': missing, 'errors': errors, 'warnings': warnings, 'files': files}


def write_audit(root: Path, result: dict) -> None:
    p = root / 'results/metrics/project_evidence_audit.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def report(root: Path) -> None:
    result = audit(root)
    write_audit(root, result)
    parts = ['# DeepHedging-VN30 — Báo cáo kết quả từ artifacts',
             f"Trạng thái bằng chứng: **{result['status']}**.",
             'Báo cáo này đọc đầu ra hiện có, không huấn luyện hoặc chọn lại mô hình. '
             'Sự hiện diện của artifacts chưa xác nhận độc lập kết quả. '
             'Episode chồng lấn; không suy ra ý nghĩa thống kê từ số episode.',
             '## Bằng chứng còn thiếu',
             '\n'.join(f'- {p}' for p in result['missing']) or 'Không thiếu file trong danh mục kiểm tra.',
             '## Lỗi và lưu ý', '\n'.join(f'- {p}' for p in result['errors'] + result['warnings']),
             '## Bảng kết quả simulator kinh tế']
    table = root / 'results/tables/full_market_simulator_frozen_summary.csv'
    if table.is_file():
        with table.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        cols = ['strategy', 'net_mean_pnl_vnd', 'net_cvar_loss_5pct_vnd',
                'mean_total_cost_vnd', 'max_margin_required_vnd']
        parts += ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
        for row in rows:
            parts.append('| ' + ' | '.join(str(row.get(c, '')).replace('|', '\\|') for c in cols) + ' |')
    else:
        parts.append('Chưa có bảng; không điền số liệu giả định vào phần kết quả.')
    parts += ['## Kết luận và giới hạn',
              'Chỉ kết luận về các metric thực sự quan sát trong bảng và giả định đã công bố. '
              'Cần đọc training curves, regime, cost và stress trước khi nghiệm thu. '
              'Không so tuyệt đối CVaR VN30 với bài toán payoff phái sinh khác. '
              'Không gọi đường trung bình trong episode là đường tài sản liên tục.',
              'Xem FINAL_REPORT.md, PRACTICAL_APPLICATION.md và METHODOLOGY_AND_LIMITATIONS.md.']
    out = root / 'results/report'
    out.mkdir(parents=True, exist_ok=True)
    text = '\n\n'.join(parts)
    (out / 'RESULTS_REPORT.md').write_text(text, encoding='utf-8')
    page = '<!doctype html><meta charset="utf-8"><title>DeepHedging-VN30 evidence</title>'
    page += '<style>body{max-width:1100px;margin:40px auto;font:16px/1.6 sans-serif;padding:20px}pre{white-space:pre-wrap}img{max-width:100%}</style>'
    page += '<h1>DeepHedging-VN30</h1><pre>' + html.escape(text) + '</pre>'
    for name in ['lstm_cvar_turnover_training_curve.png', 'real_finetune_cvar_training_curve.png',
                 'frozen_oos_mean_cumulative_performance.png', 'frozen_oos_drawdown_distribution.png',
                 'full_market_simulator_frozen_cvar.png']:
        if (root / 'results/figures' / name).is_file():
            page += f'<h2>{html.escape(name)}</h2><img src="../figures/{name}">'
    (out / 'RESULTS_REPORT.html').write_text(page, encoding='utf-8')
    print('Report:', out)


def package(root: Path, include_models: bool, allow_incomplete: bool) -> Path:
    result = audit(root)
    write_audit(root, result)
    if (result['missing'] or result['errors']) and not allow_incomplete:
        raise RuntimeError('Evidence incomplete. Inspect audit; --allow-incomplete creates a diagnostic bundle only.')
    report(root)
    selected = set()
    for folder in ['docs', 'configs', 'src', 'scripts', 'tests', 'checklist',
                   'results/metrics', 'results/tables', 'results/figures', 'results/report']:
        for p in (root / folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.name != '.gitkeep' and p.suffix != '.pyc':
                selected.add(p)
    for folder in ['data/processed', 'data/synthetic/frozen']:
        selected.update((root / folder).rglob('*.json'))
    for name in ['README.md', 'requirements.txt', '.gitignore']:
        if (root / name).is_file():
            selected.add(root / name)
    if include_models:
        selected.update((root / 'results/models').glob('*.pt'))
    # No raw/synthetic arrays, credentials, caches or logs in the default bundle.
    selected = sorted(p for p in selected if p.is_file() and not p.is_symlink()
                      and not p.name.startswith('.env') and p.suffix not in {'.key', '.pem'})
    manifest = {'audit_status': result['status'], 'includes_models': include_models,
                'files': [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size,
                           'sha256': digest(p)} for p in selected]}
    out = root / 'dist'
    out.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target = out / f'DeepHedging_VN30_{stamp}.zip'
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in selected:
            z.write(p, p.relative_to(root).as_posix())
        z.writestr('MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    print('Bundle:', target)
    return target


def evaluate(root: Path) -> None:
    prerequisites = ['results/models/lstm_cvar_turnover_best.pt',
                     'results/models/lstm_cvar_turnover_real_finetuned_best.pt',
                     'data/processed/deep_hedging_real/test_sequences_final_oos.npz',
                     'data/processed/splits/train.csv', 'data/processed/splits/validation.csv',
                     'data/processed/splits/test.csv']
    missing = [p for p in prerequisites if not (root / p).is_file()]
    if missing:
        raise RuntimeError('Missing prerequisites; restore original artifacts first: ' + ', '.join(missing))
    before = {p: digest(root / p) for p in prerequisites}
    scripts = sorted((root / 'src/evaluation').glob('*.py'))
    scripts.append(root / 'src/utils/29_capture_environment.py')
    # Step 23 creates TEST sequences and is deliberately not rerun here.
    for script in scripts:
        if script.name.startswith('23_'):
            continue
        print('Running:', script.relative_to(root), flush=True)
        subprocess.run([sys.executable, str(script)], cwd=root, check=True)
    for p, checksum in before.items():
        if digest(root / p) != checksum:
            raise RuntimeError('Frozen input changed: ' + p)
    report(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit', 'report', 'package', 'evaluate'])
    parser.add_argument('--include-models', action='store_true')
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    try:
        if args.command == 'audit':
            result = audit(ROOT)
            write_audit(ROOT, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1 if result['missing'] or result['errors'] else 0
        if args.command == 'report':
            report(ROOT)
        elif args.command == 'package':
            package(ROOT, args.include_models, args.allow_incomplete)
        else:
            evaluate(ROOT)
        return 0
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
