Runtime assets used by packaged builds.

- `uv` / `uv.exe` is downloaded at release build time by:
  - `scripts/download-uv.sh`
  - `scripts/download-uv.ps1`

The Electron app uses this bundled `uv` binary on first launch to install
Python + dependencies into the user app-data directory.
