from argparse import ArgumentParser
from pathlib import Path
import subprocess
import shutil
import tempfile


parser = ArgumentParser()
parser.add_argument("notebook", type=Path)
args = parser.parse_args()

build_dir = Path("build")

with tempfile.TemporaryDirectory() as temp_dir:
    temp_notebook = Path(temp_dir) / args.notebook.name
    temp_script = Path(temp_dir) / args.notebook.with_suffix(".py").name
    saved_script = build_dir / temp_script.name
    shutil.copy(args.notebook, temp_notebook)
    subprocess.run(["jupyter", "nbconvert", "--to", "script", temp_notebook.as_posix()])
    subprocess.run(
        [
            "stickytape",
            temp_script.as_posix(),
            "--add-python-path",
            args.notebook.parent.as_posix(),
            "--output-file",
            saved_script.as_posix(),
        ]
    )
