#!/bin/bash
# Regenerate Python source from Qt Designer .ui files.
# Run from src/main/python/ui/ via: poetry run bash convert.sh
for ui in *.ui
do
    echo "Compiling ${ui}"
    poetry run pyuic6 "${ui}" -o "${ui%.ui}.py"
done