"""Mockable bittensor wrapper.

Default (live) mode delegates to the real bittensor package.
Call `init()` before use to switch modes:

    from tau import bittensor as bt
    bt.init(mode="debug")           # log chain ops to console / file
    bt.init(mode="test")            # silent in-memory mocks; no chain
    bt.init(mode="debug", debug_output_path="/tmp/bt.jsonl")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)

_mode: Literal["live", "test", "debug"] = "live"
_debug_output_path: Path | None = None


def init(
    mode: Literal["live", "test", "debug"] = "live",
    debug_output_path: str | Path | None = None,
) -> None:
    """Switch the bittensor module mode.

    Args:
        mode: "live" uses the real chain; "debug" logs chain ops to console or
            *debug_output_path*; "test" silently succeeds with in-memory mocks.
        debug_output_path: File to append JSON-lines debug output (debug mode only).
            Defaults to console via the module logger.
    """
    global _mode, _debug_output_path
    _mode = mode
    _debug_output_path = Path(debug_output_path) if debug_output_path else None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _debug_log(data: dict[str, Any]) -> None:
    msg = json.dumps(data, indent=2, default=str)
    if _debug_output_path:
        with open(_debug_output_path, "a") as fh:
            fh.write(msg + "\n")
    else:
        log.info("[bittensor-debug]\n%s", msg)


# ---------------------------------------------------------------------------
# Mock types
# ---------------------------------------------------------------------------

class _MockWeightsResult:
    success = True
    message = "bittensor mock: weights not submitted to chain"


class _MockExtrinsics:
    def set_weights(
        self,
        wallet: Any,
        netuid: int,
        uids: Any,
        weights: Any,
        **kwargs: Any,
    ) -> _MockWeightsResult:
        if _mode == "debug":
            _debug_log({
                "action": "set_weights",
                "netuid": netuid,
                "wallet_name": getattr(wallet, "name", None),
                "wallet_hotkey": getattr(wallet, "hotkey_str", None),
                "uids": list(uids),
                "weights": [float(w) for w in weights],
            })
        return _MockWeightsResult()


class _MockSubtensor:
    def __init__(self, network: str | None = None, **kwargs: Any) -> None:
        self.extrinsics = _MockExtrinsics()
        if _mode == "debug":
            _debug_log({"action": "SubtensorApi_open", "network": network})


class _MockWallet:
    def __init__(self, name: str | None = None, hotkey: str | None = None, path: str | None = None) -> None:
        self.name = name
        self.hotkey_str = hotkey
        self.path = path


class _MockKeypair:
    def __init__(self, ss58_address: str) -> None:
        self.ss58_address = ss58_address

    def verify(self, message: Any, signature: Any) -> bool:
        return True


# ---------------------------------------------------------------------------
# Public API — mirrors real bittensor surface used by validate.py
# ---------------------------------------------------------------------------

def Wallet(name: str | None = None, hotkey: str | None = None, path: str | None = None) -> Any:
    if _mode == "live":
        import bittensor as _bt
        return _bt.Wallet(name=name, hotkey=hotkey, path=path)
    return _MockWallet(name=name, hotkey=hotkey, path=path)


def Keypair(ss58_address: str) -> Any:
    if _mode == "live":
        import bittensor as _bt
        return _bt.Keypair(ss58_address=ss58_address)
    return _MockKeypair(ss58_address=ss58_address)


def SubtensorApi(network: str | None = None, **kwargs: Any) -> Any:
    if _mode == "live":
        import bittensor as _bt
        if network is not None:
            return _bt.SubtensorApi(network=network, **kwargs)
        return _bt.SubtensorApi(**kwargs)
    return _MockSubtensor(network=network, **kwargs)
