# SPDX-License-Identifier: Apache-2.0
import os, json, pathlib

class EphapticRuntimeGuard:
    def __init__(self, agent_label: str, state_dir: str = ".ephapsys_state"):
        self.agent_label = agent_label
        self.state_dir = pathlib.Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _status_path(self):
        return self.state_dir / f"{self.agent_label}.status.json"

    def set_status(self, status: str, ecm_path: str = ""):
        with open(self._status_path(), "w") as f:
            json.dump({"status": status, "ecm_path": ecm_path}, f)

    def load_status(self):
        if self._status_path().exists():
            with open(self._status_path()) as f:
                return json.load(f)
        return {"status":"enabled","ecm_path":""}

    def allow_inference(self):
        st = self.load_status()
        if st.get("status") in ("revoked","disabled"):
            return False, f"Agent {self.agent_label} is {st.get('status')}"
        ecm = st.get("ecm_path") or os.getenv("EPHAPSYS_ECM_PATH","")
        if not ecm or not os.path.exists(ecm):
            return False, f"ECM not present for agent {self.agent_label}"
        return True, "ok"
