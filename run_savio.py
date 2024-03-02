from glob import glob
import os
import subprocess
import time
import pdb

for path in glob("savio_scripts/*.sh"):
    subprocess.run(
        [
            "sbatch",
            f"{path}",
        ]
    )
