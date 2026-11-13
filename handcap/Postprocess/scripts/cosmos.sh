#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "$script_dir/_02_combine_and_transfer_data_into_cosmos.py" "$@"
