"""Offline defaults for the copied environment; local distributed sockets work."""
from __future__ import annotations

import os
import sys


def _offline_socket_audit(event, args):
    if event != 'socket.connect':
        return
    address = args[1]
    # AF_UNIX sockets and local TCP rendezvous do not access remote services.
    if isinstance(address, (str, bytes)):
        return
    host = str(address[0]) if address else ''
    if host in ('localhost', '::1', '0.0.0.0') or host.startswith('127.'):
        return
    raise RuntimeError(
        'GRN_S5 is offline: remote Python socket connection was blocked. '
        'All required model caches should be included in protocol/offline_cache.'
    )


if os.environ.get('GRN_S5_ALLOW_NETWORK', '0') != '1':
    sys.addaudithook(_offline_socket_audit)

if os.environ.get('S5_BUNDLE_GUARD') == '1':
    import bundle_guard
    bundle_guard.install()


# Apply after the bundle guard so the policy's imports remain audited.
if os.environ.get('GRN_S5_BACKEND') == 'maca':
    from maca_numeric_policy import configure_maca_numeric_policy
    try:
        configure_maca_numeric_policy()
    except Exception as error:
        raise SystemExit('Failed to configure the MACA numerical policy: ' + str(error)) from error
