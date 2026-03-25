This folder contains helper shell scripts to build, test, and publish the Ephapsys SDK.

## Recommended Default

Most developers should start with:

```bash
source ./use-sdk.sh --testpypi
```

That is the simplest day-to-day command because it both installs or upgrades the SDK
and activates the matching virtualenv in your current shell.

## Files
- `setup.sh`: Create a dedicated virtualenv and install the SDK from the local repo, TestPyPI, or PyPI.
- `use-sdk.sh`: Sourceable wrapper around `setup.sh` that also activates the target virtualenv in your current shell.
- `publish-sdk.sh`: Build and upload the SDK to TestPyPI or PyPI. Supports:
  - `--dev` – local rebuild + reinstall (delegates to `cleanup-sdk.sh`).
  - `--stag` – build + upload to TestPyPI (auto-bumps patch unless `PUBLISH_VERSION` is set). Alias: `--staging`.
  - `--prod` – build + upload to PyPI (uses local pyproject version by default; only auto-bumps if that exact version already exists on PyPI). Prompts unless `PUBLISH_FORCE=1`. Alias: `--production`.
- `cleanup-sdk.sh`: Rebuild the wheel and reinstall locally from `sdk/python`.

## Usage
From this directory:
```bash
# Install a clean SDK test environment from TestPyPI
./setup.sh --testpypi --version 0.2.21

# Install or upgrade to the latest SDK from TestPyPI
./setup.sh --testpypi

# Install from the local repo in editable mode
./setup.sh --local

# Install/upgrade and activate in the current shell
source ./use-sdk.sh --testpypi

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

Use `setup.sh` when you only want to prepare a venv.

Use `use-sdk.sh` when you want the installed SDK to become the active `python`
and `ephapsys` in your current shell immediately after installation.

Use `publish-sdk.sh` only for SDK releases.

Use `cleanup-sdk.sh` only for local rebuild/reinstall workflows when that specific behavior is what you want.

You need valid Twine credentials:
- For TestPyPI: `~/.pypirc` with a `testpypi` section or `TWINE_API_KEY`/`TWINE_USERNAME`/`TWINE_PASSWORD`.
- For PyPI: `~/.pypirc` with a `pypi` section or `TWINE_API_KEY`/`TWINE_USERNAME`/`TWINE_PASSWORD`.
