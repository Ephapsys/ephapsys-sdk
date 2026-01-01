# SPDX-License-Identifier: Apache-2.0
class PkiClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.token = token
    def issue_model_cert(self, subject_dn: str):
        return {"serial":"DEMO1234","subject_dn":subject_dn}


def issue_model_ecm_cert(base_url: str, api_key: str, org_id: str, model_template_id: str, ecm_digest: str, job_id: str):
    from .http import request
    body = {"org_id": org_id, "model_template_id": model_template_id, "ecm_digest": ecm_digest, "job_id": job_id}
    return request(base_url.rstrip('/'), "/api/pki/model/ecm/issue", "POST", body, api_key)

    def issue_model_ecm_cert(self, org_id: str, model_template_id: str, rms_hash: str, job_id: str | None = None, extra: dict | None = None):
        body = {"org_id": org_id, "model_template_id": model_template_id, "rms_hash": rms_hash}
        if job_id: body["job_id"] = job_id
        if extra: body["extra"] = extra
        return request(self.base_url, "/api/api/pki/model_ecm_cert", "POST", body, self.api_key)
