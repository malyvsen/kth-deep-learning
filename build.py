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

with saved_script.open("r") as script:
    code = script.read()

with saved_script.open("w") as script:
    script.write(
        "# This file is auto-generated from multiple files, because my code is a package.\n"
    )
    script.write(
        "# To look at the code the way it's meant to be looked at, visit https://github.com/malyvsen/kth-deep-learning\n"
    )
    script.write(code)
