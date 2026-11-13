"""Audit hook proving package runs cannot fall back to the original project."""
import atexit
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
_INSTALLED = False
_BLOCKED = []
_READ_ROOTS = set()
FORBIDDEN = (
    '/inspire/store/project/fdu-nicu/public/hanzhiyong/GRN',
    '/inspire/store/project/fdu-nicu/public/hanzhiyong/VARSR-AlignVAR-Offline4GPU',
    '/inspire/store/project/fdu-nicu/public/hanzhiyong/AlignVAR-Offline4GPU',
    '/tmp/grn_sr_eval_venv',
)

def install():
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = os.path.abspath(os.fsdecode(args[0]))
        # lexical comparison avoids performing further filesystem calls inside an audit hook
        if any(path == base or path.startswith(base + '/') for base in FORBIDDEN):
            _BLOCKED.append(path)
            raise PermissionError('GRN_S5 isolation: attempted access to original project: ' + path)
        if path.startswith(str(ROOT) + '/'):
            _READ_ROOTS.add(path[len(str(ROOT))+1:].split('/')[0])

    sys.addaudithook(audit)
    report_dir = os.environ.get('S5_GUARD_REPORT_DIR')
    if report_dir:
        report_dir = Path(report_dir)
        if not report_dir.resolve().is_relative_to(ROOT):
            raise ValueError('S5_GUARD_REPORT_DIR must be inside the bundle')
        report_dir.mkdir(parents=True, exist_ok=True)

        def report():
            path = report_dir / ('process_' + str(os.getpid()) + '.json')
            path.write_text(json.dumps(dict(
                status='passed' if not _BLOCKED else 'blocked_external_access',
                pid=os.getpid(), argv=sys.argv, forbidden_roots=list(FORBIDDEN),
                attempted_external_access=_BLOCKED, read_bundle_sections=sorted(_READ_ROOTS),
            ), indent=2, ensure_ascii=False) + '\n')
        atexit.register(report)
