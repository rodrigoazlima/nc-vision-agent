import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "src" / "nc_vision_agent" / "tools" / "classify_images.py"
    sys.exit(subprocess.call([sys.executable, str(script), *sys.argv[1:]]))
