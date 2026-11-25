"""Google OAuth for the Gmail and Calendar MCP servers.

``login`` runs once, interactively (``assistant google-login``): the browser consent
flow with a Desktop OAuth client, asking for offline access so Google issues a refresh
token. The refresh token is kept in the OS keychain (or a 0600 file on headless
devices); at runtime ``GoogleTokenProvider`` trades it for short-lived access tokens.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Literal, Optional, Protocol

import httpx
from pydantic import BaseModel, ConfigDict

TOKEN_URL = "https://oauth2.googleapis.com/token"
TOKEN_HOST = "oauth2.googleapis.com"
DEFAULT_SCOPES = (
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.events",
)
_KEYRING_SERVICE = "assistant-iot"
_KEYRING_USER = "google-oauth"


class GoogleOAuthSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_secrets: Path = Path("~/.config/assistant-iot/google-client-secret.json")
    """The Desktop-app OAuth client JSON downloaded from Google Cloud Console."""
    token_store: Literal["keyring", "file"] = "keyring"
    token_file: Path = Path("~/.config/assistant-iot/google-token.json")
    scopes: tuple[str, ...] = DEFAULT_SCOPES


class StoredCredentials(BaseModel):
    client_id: str
    client_secret: str
    refresh_token: str
    scopes: tuple[str, ...]


class CredentialStore(Protocol):
    def load(self) -> Optional[StoredCredentials]: ...

    def save(self, credentials: StoredCredentials) -> None: ...


class KeyringStore:
    def load(self) -> Optional[StoredCredentials]:
        import keyring

        raw = keyring.get_password(_KEYRING_SERVICE, _KEYRING_USER)
        return StoredCredentials.model_validate_json(raw) if raw else None

    def save(self, credentials: StoredCredentials) -> None:
        import keyring

        keyring.set_password(_KEYRING_SERVICE, _KEYRING_USER, credentials.model_dump_json())


class FileStore:
    """For headless devices with no keychain. The file is created readable by the owner only."""

    def __init__(self, path: Path) -> None:
        self._path = path.expanduser()

    def load(self) -> Optional[StoredCredentials]:
        if not self._path.exists():
            return None
        return StoredCredentials.model_validate_json(self._path.read_text())

    def save(self, credentials: StoredCredentials) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(credentials.model_dump_json())


def make_store(settings: GoogleOAuthSettings) -> CredentialStore:
    return KeyringStore() if settings.token_store == "keyring" else FileStore(settings.token_file)


def login(settings: GoogleOAuthSettings) -> StoredCredentials:
    """Interactive consent in the browser; stores and returns the credentials."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(settings.client_secrets.expanduser()), scopes=list(settings.scopes))
    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent", open_browser=True)
    if not creds.refresh_token:
        raise RuntimeError("Google did not return a refresh token; is the OAuth client a Desktop app?")
    stored = StoredCredentials(
        client_id=creds.client_id,
        client_secret=creds.client_secret,
        refresh_token=creds.refresh_token,
        scopes=tuple(creds.scopes or settings.scopes),
    )
    make_store(settings).save(stored)
    return stored


class NotLoggedIn(RuntimeError):
    pass


class GoogleTokenProvider:
    """Returns a valid access token, refreshing it shortly before it expires."""

    _MARGIN_SECONDS = 60

    def __init__(self, store: CredentialStore, client: httpx.AsyncClient, clock: Callable[[], float] = time.monotonic) -> None:
        self._store = store
        self._client = client
        self._clock = clock
        self._token: Optional[str] = None
        self._expires_at = 0.0

    async def __call__(self) -> str:
        if self._token is None or self._clock() >= self._expires_at - self._MARGIN_SECONDS:
            await self._refresh()
        assert self._token is not None
        return self._token

    async def _refresh(self) -> None:
        credentials = self._store.load()
        if credentials is None:
            raise NotLoggedIn("no Google credentials stored; run: poetry run assistant google-login")
        response = await self._client.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "client_id": credentials.client_id,
                "client_secret": credentials.client_secret,
                "refresh_token": credentials.refresh_token,
            },
        )
        if response.status_code != 200:
            error = response.json().get("error", response.status_code) if response.content else response.status_code
            raise NotLoggedIn(f"Google token refresh failed ({error}); run: poetry run assistant google-login")
        body = response.json()
        self._token = body["access_token"]
        self._expires_at = self._clock() + float(body.get("expires_in", 3600))
