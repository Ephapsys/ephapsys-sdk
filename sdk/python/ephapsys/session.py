# SPDX-License-Identifier: Apache-2.0
import os, json, requests, getpass
from pathlib import Path

# Where session state is stored
STATE_FILE = Path.home() / ".ephapsys_state" / "session.json"
DEFAULT_API_URL = "https://api.ephapsys.com"


def get_api_url(base_url: str = None) -> str:
    return (
        base_url
        or os.getenv("AOC_BASE_URL")
        or os.getenv("AOC_API_URL")
        or os.getenv("EPHAPSYS_API_URL")
        or DEFAULT_API_URL
    )


API_URL = get_api_url()

def _ensure_state_dir():
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

def load_session():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_session(data: dict):
    _ensure_state_dir()
    STATE_FILE.write_text(json.dumps(data, indent=2))

def get_token():
    sess = load_session()
    return sess.get("token")

def login(username: str = None, password: str = None, base_url: str = None) -> str:
    """
    Authenticate against /cli/login and store JWT in session.json.
    """
    api_url = get_api_url(base_url)
    username = username or input("Username: ")
    password = password or getpass.getpass("Password: ")
    try:
        resp = requests.post(f"{api_url}/cli/login", json={
            "username": username,
            "password": password,
        })
    except requests.RequestException as e:
        raise SystemExit(f"❌ Failed to reach {api_url}/cli/login: {e}")

    if resp.status_code != 200:
        detail = ""
        try:
            detail = resp.json().get("detail") or ""
        except Exception:
            pass
        msg = f"❌ Login failed ({resp.status_code})."
        if detail:
            msg += f" {detail}"
        raise SystemExit(msg)

    data = resp.json()
    data["base_url"] = api_url
    save_session(data)
    print(f"✅ Logged in as {username}")
    return data["token"]

def auth_headers() -> dict:
    """
    Return Authorization header for API calls.
    If no valid token is found, exit with message.
    """
    token = get_token()
    if not token:
        raise SystemExit("⚠️ No active session. Run `ephapsys login` first.")
    return {"Authorization": f"Bearer {token}"}
