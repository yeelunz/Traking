from __future__ import annotations
import os
import platform
import subprocess
from typing import Dict, Optional


def capture_env(skip_freeze: bool | None = None, timeout: float = 8.0) -> Dict[str, str]:
    """Collect lightweight環境資訊。

    為避免在 Windows / 防毒 或大量套件環境下 `pip freeze` 阻塞 UI：
    - 可透過參數或環境變數 `TRACKING_SKIP_PIP_FREEZE=1` 跳過。
    - 增設 timeout，逾時不影響主流程。
    """
    info: Dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if skip_freeze is None:
        skip_freeze = os.environ.get("TRACKING_SKIP_PIP_FREEZE", "0") in ("1", "true", "True")
    if skip_freeze:
        return info
    try:
        # 使用 run + timeout，避免永久卡住
        proc = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout)),
            check=False,
        )
        if proc.stdout:
            # 若輸出過大，可截斷（避免 metadata.json 體積爆炸）
            out = proc.stdout
            if len(out) > 60_000:  # ~60KB 上限
                out = out[:60_000] + "\n# -- truncated --"
            info["pip_freeze"] = out
    except subprocess.TimeoutExpired:
        info["pip_freeze"] = "<timeout>"
    except Exception as e:  # pragma: no cover - 安全防護
        info["pip_freeze_error"] = f"{type(e).__name__}: {e}"
    return info
