"""
DEX session authentication for Kubeflow 1.11.
Lifted directly from ev_models_inference/kf_helper.py.
"""
import re
import time
from urllib.parse import urlsplit, urlencode

import requests
import urllib3


class DexSessionManager:
    """
    Authenticates against a Kubeflow DEX endpoint and returns session cookies
    that can be passed directly to the KFP Client.

    See: https://www.kubeflow.org/docs/components/pipelines/user-guides/core-functions/connect-api/#kubeflow-platform---outside-the-cluster
    """

    def __init__(
        self,
        endpoint_url: str,
        dex_username: str,
        dex_password: str,
        dex_auth_type: str = "local",
        skip_tls_verify: bool = False,
    ):
        self._endpoint_url = endpoint_url
        self._skip_tls_verify = skip_tls_verify
        self._dex_username = dex_username
        self._dex_password = dex_password
        self._dex_auth_type = dex_auth_type

        if self._skip_tls_verify:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        if self._dex_auth_type not in ["ldap", "local"]:
            raise ValueError(
                f"Invalid dex_auth_type '{self._dex_auth_type}', must be one of: ['ldap', 'local']"
            )

    def get_session_cookies(self) -> str:
        """
        Authenticate against DEX and return cookies as "key1=value1; key2=value2".
        Retries up to 3 times with exponential backoff.
        """
        max_retries = 3
        base_retry_delay = 2

        for attempt in range(max_retries):
            session = requests.Session()

            if attempt > 0:
                time.sleep(base_retry_delay * (2 ** (attempt - 1)))

            try:
                response = session.get(
                    self._endpoint_url,
                    allow_redirects=True,
                    verify=not self._skip_tls_verify,
                )

                if response.status_code in [401, 403]:
                    url_object = urlsplit(response.url)
                    url_object = url_object._replace(
                        path="/oauth2/start",
                        query=urlencode({"rd": url_object.path}),
                    )
                    response = session.get(
                        url_object.geturl(),
                        allow_redirects=True,
                        verify=not self._skip_tls_verify,
                    )
                    if response.status_code not in [200, 302]:
                        raise RuntimeError(
                            f"HTTP {response.status_code} on GET /oauth2/start"
                        )
                elif response.status_code != 200:
                    raise RuntimeError(
                        f"HTTP {response.status_code} on GET {self._endpoint_url}"
                    )

                if len(response.history) == 0:
                    return ""

                url_object = urlsplit(response.url)
                if re.search(r"/auth$", url_object.path):
                    url_object = url_object._replace(
                        path=re.sub(r"/auth$", f"/auth/{self._dex_auth_type}", url_object.path)
                    )

                if re.search(r"/auth/.*/login$", url_object.path):
                    dex_login_url = url_object.geturl()
                else:
                    response = session.get(
                        url_object.geturl(),
                        allow_redirects=True,
                        verify=not self._skip_tls_verify,
                    )
                    if response.status_code != 200:
                        raise RuntimeError(
                            f"HTTP {response.status_code} on GET {url_object.geturl()}"
                        )
                    dex_login_url = response.url

                response = session.post(
                    dex_login_url,
                    data={"login": self._dex_username, "password": self._dex_password},
                    allow_redirects=True,
                    verify=not self._skip_tls_verify,
                )

                if response.status_code == 403:
                    oauth_url = (
                        f"{urlsplit(self._endpoint_url).scheme}://"
                        f"{urlsplit(self._endpoint_url).netloc}/oauth2/start"
                    )
                    response = session.get(oauth_url, allow_redirects=True, verify=not self._skip_tls_verify)
                    if response.status_code == 200 and session.cookies:
                        return "; ".join([f"{c.name}={c.value}" for c in session.cookies])

                if response.status_code != 200:
                    raise RuntimeError(
                        f"HTTP {response.status_code} on POST {dex_login_url}"
                    )

                if len(response.history) == 0:
                    raise RuntimeError(
                        f"No redirect after POST to {dex_login_url} — credentials may be invalid"
                    )

                url_object = urlsplit(response.url)
                if re.search(r"/approval$", url_object.path):
                    response = session.post(
                        url_object.geturl(),
                        data={"approval": "approve"},
                        allow_redirects=True,
                        verify=not self._skip_tls_verify,
                    )
                    if response.status_code != 200:
                        raise RuntimeError(
                            f"HTTP {response.status_code} on POST /approval"
                        )

                return "; ".join([f"{c.name}={c.value}" for c in session.cookies])

            except Exception:
                if attempt == max_retries - 1:
                    raise
