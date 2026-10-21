#!/usr/bin/env bash
# ---------------------------------------------------------------
# wsl-build.sh - the Linux half of the compile step for this project.
#
# Called by build.bat with a single argument: the absolute
# /mnt/c/... path of the .tex file to compile.
#
# Do not call this directly from Windows. Use build.bat, which
# handles the path translation.
# ---------------------------------------------------------------

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: wsl-build.sh /mnt/c/path/to/file.tex" >&2
    exit 1
fi

tex_path="$1"
tex_dir=$(dirname "$tex_path")
tex_file=$(basename "$tex_path")

if ! command -v latexmk >/dev/null 2>&1; then
    echo "latexmk not found in WSL. Install with:" >&2
    echo "    sudo apt install texlive-full latexmk" >&2
    exit 1
fi

cd "$tex_dir"
mkdir -p build

# -outdir=build keeps the output next to (not on top of) the source.
latexmk \
    -pdf \
    -interaction=nonstopmode \
    -halt-on-error \
    -outdir=build \
    "$tex_file"

echo
echo "Done. PDF at: $tex_dir/build/${tex_file%.tex}.pdf"
