from __future__ import annotations

import threading
from dataclasses import dataclass, field

import requests

from .config import API_TIMEOUT_SECONDS, API_URL
from .model import get_model


@dataclass
class PredictionBatch:
    values: list[float]
    unseen_inputs: list[list[str]]
    served_by: str
    note: str | None = None
    api_url: str | None = None
    errors: list[str] = field(default_factory=list)


class InferenceClient:
    def __init__(self, api_url: str | None = API_URL, timeout: float = API_TIMEOUT_SECONDS):
        self.api_url = (api_url or "").rstrip("/") or None
        self.timeout = timeout

    def predict(self, records: list[dict[str, str]]) -> PredictionBatch:
        note = None
        if self.api_url:
            try:
                return self._predict_via_api(records)
            except (requests.RequestException, ValueError, KeyError) as exc:
                note = (
                    f"The inference API at {self.api_url} could not be reached "
                    f"({type(exc).__name__}), so the bundled copy of the same model was used."
                )
        return self._predict_locally(records, note)

    def _predict_via_api(self, records: list[dict[str, str]]) -> PredictionBatch:
        response = requests.post(f"{self.api_url}/predict/batch", json={"records": records}, timeout=self.timeout)
        if response.status_code == 404:
            return self._predict_one_by_one(records)
        response.raise_for_status()
        payload = response.json()["predictions"]
        return PredictionBatch(
            values=[float(item["predicted_prevalence"]) for item in payload],
            unseen_inputs=[list(item.get("unseen_inputs", [])) for item in payload],
            served_by="api",
            api_url=self.api_url,
        )

    def _predict_one_by_one(self, records: list[dict[str, str]]) -> PredictionBatch:
        values, unseen = [], []
        with requests.Session() as session:
            for record in records:
                response = session.post(f"{self.api_url}/predict", json=record, timeout=self.timeout)
                response.raise_for_status()
                body = response.json()
                values.append(float(body["predicted_prevalence"]))
                unseen.append(list(body.get("unseen_inputs", [])))
        return PredictionBatch(values=values, unseen_inputs=unseen, served_by="api", api_url=self.api_url)

    @staticmethod
    def _predict_locally(records: list[dict[str, str]], note: str | None) -> PredictionBatch:
        model = get_model()
        return PredictionBatch(
            values=model.predict_records(records),
            unseen_inputs=[model.unseen_inputs(record) for record in records],
            served_by="local",
            note=note,
        )


def get_client() -> InferenceClient:
    return InferenceClient()


def warm_up_api(api_url: str | None = API_URL, timeout: float = 90.0) -> threading.Thread | None:
    if not api_url:
        return None

    def ping() -> None:
        try:
            requests.get(f"{api_url.rstrip('/')}/health", timeout=timeout)
        except requests.RequestException:
            pass

    thread = threading.Thread(target=ping, name="lthc-api-warm-up", daemon=True)
    thread.start()
    return thread
