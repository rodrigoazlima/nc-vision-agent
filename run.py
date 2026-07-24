import logging
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_LOG_FILE = _ROOT / "state" / "logs" / "run-debug.log"
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(_LOG_FILE, encoding="utf-8")],
)
log = logging.getLogger("nc-vision-run")

if __name__ == "__main__":
    script = _ROOT / "src" / "nc_vision_agent" / "tools" / "classify_images.py"
    log.debug("dispatch: executable=%s script=%s argv=%s", sys.executable, script, sys.argv[1:])
    rc = subprocess.call([sys.executable, str(script), *sys.argv[1:]])
    log.debug("dispatch: subprocess exited with code=%d", rc)
    sys.exit(rc)
