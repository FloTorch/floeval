"""Minimal HTTP utilities for FloTorch SDK (agent config fetch, LLM calls)."""

import httpx
from typing import Any, Dict, Optional, Union

JSONType = Union[Dict[str, Any], list]


class APIError(Exception):
    """Exception raised for API response errors."""

    def __init__(self, status_code: int, message: str):
        super().__init__(f"[{status_code}] {message}")
        self.status_code = status_code
        self.message = message


def _parse_response(resp: httpx.Response) -> Any:
    """Parse HTTP response to JSON or text."""
    if 200 <= resp.status_code < 300:
        try:
            return resp.json()
        except ValueError:
            return resp.text
    raise APIError(resp.status_code, resp.text)


def http_get(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
) -> Any:
    """Send a GET request and return parsed response."""
    with httpx.Client() as client:
        resp = client.get(url, headers=headers, params=params, timeout=timeout)
        return _parse_response(resp)


def http_post(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    json: Optional[JSONType] = None,
    timeout: Optional[float] = None,
) -> Any:
    """Send a POST request with JSON body."""
    with httpx.Client(follow_redirects=True) as client:
        resp = client.post(url, headers=headers, json=json, timeout=timeout)
        return _parse_response(resp)


def http_put(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    json: Optional[JSONType] = None,
    timeout: Optional[float] = None,
) -> Any:
    """Send a PUT request with JSON body."""
    with httpx.Client() as client:
        resp = client.put(url, headers=headers, json=json, timeout=timeout)
        return _parse_response(resp)


def http_delete(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
) -> Any:
    """Send a DELETE request."""
    with httpx.Client() as client:
        resp = client.delete(url, headers=headers, timeout=timeout)
        return _parse_response(resp)


async def async_http_get(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
) -> Any:
    """Send an async GET request."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers, params=params, timeout=timeout)
        return _parse_response(resp)


async def async_http_post(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    json: Optional[JSONType] = None,
    timeout: Optional[float] = None,
) -> Any:
    """Send an async POST request with JSON body."""
    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.post(url, headers=headers, json=json, timeout=timeout)
        return _parse_response(resp)


async def async_http_put(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    json: Optional[JSONType] = None,
    timeout: Optional[float] = None,
) -> Any:
    """Send an async PUT request with JSON body."""
    async with httpx.AsyncClient() as client:
        resp = await client.put(url, headers=headers, json=json, timeout=timeout)
        return _parse_response(resp)


async def async_http_delete(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
) -> Any:
    """Send an async DELETE request."""
    async with httpx.AsyncClient() as client:
        resp = await client.delete(url, headers=headers, timeout=timeout)
        return _parse_response(resp)
