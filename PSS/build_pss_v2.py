"""Build the shared C++ implementation used by Python and R."""
import os
from pathlib import Path
import subprocess
import sys

here=Path(__file__).resolve().parent
mac=sys.platform=="darwin"
subprocess.run([os.environ.get("CXX","c++"),"-O3","-std=c++17",
                *(["-dynamiclib"] if mac else ["-shared","-fPIC"]),
                str(here/"pss_v2.cpp"),"-o",str(here/("libpss_v2.dylib" if mac else "libpss_v2.so"))],check=True)
