from subprocess import call
import sys

rc = call(["python3", "compile.py", "build_ext", "--inplace"])
if rc != 0:
    sys.exit(rc)

# Needs to be done after the above call
import learn
learn.run(sys.argv[1:])
