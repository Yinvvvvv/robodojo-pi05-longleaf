# Third-party sources

The files under `upstream-overlay/XPolicyLab/policy/Pi_05/openpi/` come from the XPolicyLab openpi integration pinned to commit `53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2` and the preserved Longleaf runtime snapshot. Their Apache-2.0 license is retained in that directory. Original notices in their bodies are preserved.

The runtime `scripts/train.py`, `training/checkpoints.py` and `training/optimizer.py` are exported from the verified local capture, including any runtime checkpoint-policy adaptations. A leading export-provenance comment is the only additional modification made by this export; `SOURCE_MANIFEST.json` records the digest of the exact captured runtime body. The base `training/config.py` is copied from the pinned upstream revision.

This private archive does not assign a new public license to the project's own Longleaf scripts. Refer to upstream licenses for the third-party components.
