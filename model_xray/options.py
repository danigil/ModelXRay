"""Module-level constants resolved from environment variables.

Historical entrypoint for older library code. Most filesystem paths now also
appear under `model_xray.data.paths` with a function API; the constants here
are kept for the canonical procedures (`procedures.cover_data_procs`,
`procedures.embedding_procs`, `utils.model_utils`) which read them by name.

Environment variables:
  MODELXRAY_GHRP_DIR              : GHRP zoo root (D1, D5)
  MODELXRAY_MALEFICNET_DIR        : MaleficNet attacked-image dir (D4)
  MODELXRAY_MALEFICNET_DOWNLOADS  : MaleficNet auxiliary download cache (D4 attack regen)
  MODELXRAY_MALEFICNET_PAYLOADS   : MaleficNet payload binaries dir (D4 attack regen)
  HF_HOME                         : Hugging Face cache; defaults to ~/.cache/huggingface
"""

from __future__ import annotations

import os


_UNSET = "/SET_THIS_ENV_VAR"

GHRP_MZS_DIR = os.environ.get("MODELXRAY_GHRP_DIR", _UNSET + "/MODELXRAY_GHRP_DIR")
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

MALEFICNET_DATASET_DOWNLOAD_DIR = os.environ.get(
    "MODELXRAY_MALEFICNET_DOWNLOADS", _UNSET + "/MODELXRAY_MALEFICNET_DOWNLOADS"
)
MALEFICNET_PAYLOADS_DIR = os.environ.get(
    "MODELXRAY_MALEFICNET_PAYLOADS", _UNSET + "/MODELXRAY_MALEFICNET_PAYLOADS"
)


def get_maleficnet_payload_path(mal_name: str) -> str:
    return os.path.join(MALEFICNET_PAYLOADS_DIR, mal_name)
