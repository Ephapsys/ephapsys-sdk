This folder contains helper shell scripts to build, test, and publish the Ephapsys SDK.

## Files
- `publish-sdk.sh`: Build and upload the SDK to TestPyPI or PyPI. Supports:
  - `--dev` – local rebuild + reinstall (delegates to `cleanup-sdk.sh`).
  - `--stag` – build + upload to TestPyPI (auto-bumps patch unless `PUBLISH_VERSION` is set). Alias: `--staging`.
  - `--prod` – build + upload to PyPI (uses local pyproject version by default; only auto-bumps if that exact version already exists on PyPI). Prompts unless `PUBLISH_FORCE=1`. Alias: `--production`.
- `cleanup-sdk.sh`: Rebuild the wheel and reinstall locally from `sdk/python`.

## Usage
From this directory:
```bash
# Local rebuild/install
./publish-sdk.sh --dev

# Publish to TestPyPI (auto-bumps patch)
./publish-sdk.sh --stag

# Publish to PyPI (prompts for confirmation)
./publish-sdk.sh --prod
```

Optional environment variables / flags:
- `--verbose` flag – show full pip output (default is quieter installs).
- `SDK_DIR` – override the SDK path (defaults to `../sdk/python`).
- `PUBLISH_VERSION` – force a specific version instead of auto-bumping.
- `PUBLISH_VERSION` – force a specific version for either `--stag` or `--prod`.
- `PUBLISH_FORCE=1` – skip the confirmation prompt for production uploads.

You need valid Twine credentials:
- For TestPyPI: `~/.pypirc` with a `testpypi` section or `TWINE_API_KEY`/`TWINE_USERNAME`/`TWINE_PASSWORD`.
- For PyPI: `~/.pypirc` with a `pypi` section or `TWINE_API_KEY`/`TWINE_USERNAME`/`TWINE_PASSWORD`.
