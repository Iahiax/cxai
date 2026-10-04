# core/session.py
from __future__ import annotations
import threading, time
from dataclasses import dataclass, field
from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class CapitalSession:
    api_key: str
    identifier: str
    password: str
    demo: bool = True

    cst: Optional[str] = field(default=None, init=False, repr=False)
    xst: Optional[str] = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _last_refresh: float = field(default=0.0, init=False, repr=False)
    _http: requests.Session = field(init=False, repr=False)

    def __post_init__(self):
        self.base_url = (
            "https://demo-api-capital.backend-capital.com"
            if self.demo else
            "https://api-capital.backend-capital.com"
        )
        retry = Retry(
            total=3, backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST", "PUT", "DELETE"]),
        )
        s = requests.Session()
        s.mount("https://", HTTPAdapter(max_retries=retry))
        self._http = s

    def login(self) -> None:
        with self._lock:
            url = f"{self.base_url}/api/v1/session"
            headers = {"X-CAP-API-KEY": self.api_key, "Content-Type": "application/json"}
            payload = {
                "identifier": self.identifier,
                "password": self.password,
                "encryptedPassword": False,
            }
            r = self._http.post(url, headers=headers, json=payload, timeout=15)
            if r.status_code != 200:
                raise RuntimeError(f"Login failed {r.status_code}: {r.text}")
            self.cst = r.headers.get("CST")
            self.xst = r.headers.get("X-SECURITY-TOKEN")
            if not self.cst or not self.xst:
                raise RuntimeError("Login succeeded but tokens missing in headers")
            self._last_refresh = time.time()

    def refresh(self) -> None:
        with self._lock:
            if not self.cst or not self.xst:
                return self.login()
            url = f"{self.base_url}/api/v1/session/refresh"
            headers = {
                "X-CAP-API-KEY": self.api_key,
                "CST": self.cst,
                "X-SECURITY-TOKEN": self.xst,
            }
            r = self._http.post(url, headers=headers, timeout=15)
            if r.status_code != 200:
                return self.login()
            self.cst = r.headers.get("CST", self.cst)
            self.xst = r.headers.get("X-SECURITY-TOKEN", self.xst)
            self._last_refresh = time.time()

    def auth_headers(self) -> dict:
        # جلسة Capital صالحة ~10 دقائق → نجدّد كل 8 دقائق
        if time.time() - self._last_refresh > 8 * 60:
            self.refresh()
        return {
            "X-CAP-API-KEY": self.api_key,
            "CST": self.cst or "",
            "X-SECURITY-TOKEN": self.xst or "",
            "Content-Type": "application/json",
        }

    def request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self.base_url}{path}"
        r = self._http.request(method, url, headers=self.auth_headers(), timeout=20, **kwargs)
        if r.status_code == 401:
            self.refresh()
            r = self._http.request(method, url, headers=self.auth_headers(), timeout=20, **kwargs)
        r.raise_for_status()
        return r
