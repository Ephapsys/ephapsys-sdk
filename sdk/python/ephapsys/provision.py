# SPDX-License-Identifier: Apache-2.0
import json, pathlib, time

class Provisioner:
    def __init__(self, state_dir: str = ".ephapsys_state"):
        self.state_dir = pathlib.Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def bind(self, agent_label: str, device_id: str, ecm_path: str) -> str:
        rec = {
            "agent_label": agent_label,
            "device_id": device_id,
            "ecm_path": ecm_path,
            "bound_at": int(time.time()),
        }
        path = self.state_dir / f"{agent_label}.binding.json"
        with open(path, "w") as f:
            json.dump(rec, f)
        return str(path)
