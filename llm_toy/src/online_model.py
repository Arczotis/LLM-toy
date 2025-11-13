"""
Online LLM client with OpenAI-compatible providers (OpenRouter, SiliconFlow).
Provides a simple wrapper with the same surface as our learning models:
- generate_text(prompt, max_length=..., temperature=..., top_p=...)
- get_model_info()

Configuration is read from llm_toy/configs/llm_api_config.json
and/or environment variables:
  - OPENROUTER_API_KEY
  - SILICONFLOW_API_KEY
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests


_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "llm_api_config.json"


def _load_api_config() -> Dict[str, Any]:
    """Load provider config from JSON with env var fallback.

    Returns a dict with shape:
    {
      "provider": "openrouter" | "siliconflow",
      "openrouter": {"api_key": str, "base_url": str, "default_model": str},
      "siliconflow": {"api_key": str, "base_url": str, "default_model": str}
    }
    """
    # Defaults if the file is missing
    cfg: Dict[str, Any] = {
        "provider": "openrouter",
        "openrouter": {
            "api_key": os.getenv("OPENROUTER_API_KEY", ""),
            "base_url": "https://openrouter.ai/api/v1",
            "default_model": "openrouter/auto",
        },
        "siliconflow": {
            "api_key": os.getenv("SILICONFLOW_API_KEY", ""),
            "base_url": "https://api.siliconflow.cn/v1",
            "default_model": "",
        },
    }

    if _CONFIG_PATH.exists():
        try:
            with _CONFIG_PATH.open("r", encoding="utf-8") as f:
                file_cfg = json.load(f)
            # Shallow merge
            for k, v in file_cfg.items():
                if isinstance(v, dict) and k in cfg:
                    cfg[k].update(v)  # type: ignore[arg-type]
                else:
                    cfg[k] = v
        except Exception as e:
            raise RuntimeError(f"Failed to read config at {_CONFIG_PATH}: {e}")

    return cfg


class OpenAICompatibleClient:
    """Minimal client for OpenAI-compatible /chat/completions providers."""

    def __init__(self, base_url: str, api_key: str, extra_headers: Optional[Dict[str, str]] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        self.extra_headers = extra_headers or {}

    def chat_completions(self, *, model: str, messages: list[dict], temperature: float = 0.7,
                         top_p: float = 1.0, max_tokens: int = 256, stream: bool = False) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        resp = self.session.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code >= 300:
            raise RuntimeError(f"Provider error {resp.status_code}: {resp.text}")
        return resp.json()


def _build_client(provider: str, cfg: Dict[str, Any]) -> tuple[OpenAICompatibleClient, str]:
    provider = provider.lower()
    if provider not in {"openrouter", "siliconflow"}:
        raise ValueError("provider must be 'openrouter' or 'siliconflow'")

    if provider == "openrouter":
        api_key = (cfg.get("openrouter") or {}).get("api_key", "")
        base_url = (cfg.get("openrouter") or {}).get("base_url", "https://openrouter.ai/api/v1")
        if not api_key or api_key.startswith("CHANGE_ME"):
            api_key = os.getenv("OPENROUTER_API_KEY", "")
        # Optional but recommended headers per OpenRouter docs
        headers = {
            "HTTP-Referer": os.getenv("OPENROUTER_APP_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "JupyterProject LLM Toy"),
        }
        return OpenAICompatibleClient(base_url, api_key, headers), provider

    # siliconflow
    api_key = (cfg.get("siliconflow") or {}).get("api_key", "")
    base_url = (cfg.get("siliconflow") or {}).get("base_url", "https://api.siliconflow.cn/v1")
    if not api_key or api_key.startswith("CHANGE_ME"):
        api_key = os.getenv("SILICONFLOW_API_KEY", "")
    return OpenAICompatibleClient(base_url, api_key), provider


class OnlineChatModel:
    """Simple online model wrapper with a generate_text API."""

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        cfg = _load_api_config()
        self.provider = (provider or cfg.get("provider") or "openrouter").lower()
        self.client, self.provider = _build_client(self.provider, cfg)

        # Choose model
        if model:
            self.model = model
        else:
            defaults = (cfg.get(self.provider) or {})
            self.model = defaults.get("default_model") or ""

        if not self.client.api_key:
            raise RuntimeError(
                f"Missing API key for provider '{self.provider}'. Set it in {_CONFIG_PATH} or via env var."
            )
        if not self.model:
            raise RuntimeError(
                f"No default model configured for provider '{self.provider}'. Set 'default_model' in {_CONFIG_PATH} "
                f"or pass model explicitly."
            )

        self.system_prompt = system_prompt or "You are a helpful assistant."

    def generate_text(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 1.0,
        do_sample: bool = True,  # kept for API parity; mapped to temperature/top_p
    ) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        data = self.client.chat_completions(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max(1, int(max_length)),
            stream=False,
        )

        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Unexpected response format: {e}; payload keys: {list(data.keys())}")

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model_name": self.model,
            "mode": "ONLINE",
            "api_base": self.client.base_url,
        }


def create_online_model(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> OnlineChatModel:
    """Factory to create an OnlineChatModel using config/env."""
    return OnlineChatModel(model=model, provider=provider, system_prompt=system_prompt)


class SmartTextModel:
    """
    Try online provider first; on any provider/network error, fall back to
    Hugging Face GPT-2 (if available) or the offline demo model.
    Maintains the same generate_text/get_model_info API shape.
    """

    def __init__(self, model: Optional[str] = None, provider: Optional[str] = None, system_prompt: Optional[str] = None):
        self._online_err: Optional[str] = None
        self._used_fallback = False
        self._fallback_model = None  # created lazily
        try:
            self._online = OnlineChatModel(model=model, provider=provider, system_prompt=system_prompt)
        except Exception as e:
            # Creation failure (e.g., missing API key). Defer to fallback on first call.
            self._online = None
            self._online_err = str(e)

    def _ensure_fallback(self):
        if self._fallback_model is None:
            # Import here to avoid circular import at module load time
            from .offline_model import create_model as create_local_model

            # Try HF GPT-2 then offline demo
            try:
                self._fallback_model = create_local_model("gpt2", force_offline=False)
            except Exception:
                # Last resort: forced offline demo
                self._fallback_model = create_local_model("gpt2", force_offline=True)

    def generate_text(self, prompt: str, max_length: int = 200, temperature: float = 0.7, top_p: float = 1.0, do_sample: bool = True) -> str:
        if self._online is not None and not self._used_fallback:
            try:
                return self._online.generate_text(prompt, max_length=max_length, temperature=temperature, top_p=top_p, do_sample=do_sample)
            except Exception as e:
                # Record and fall back
                self._online_err = str(e)
                self._used_fallback = True

        self._ensure_fallback()
        return self._fallback_model.generate_text(prompt, max_length=max_length, temperature=temperature, do_sample=do_sample)  # type: ignore[union-attr]

    def get_model_info(self) -> Dict[str, Any]:
        if not self._used_fallback and self._online is not None:
            info = self._online.get_model_info()
            if self._online_err:
                info = {**info, "warning": f"online init note: {self._online_err}"}
            return info
        self._ensure_fallback()
        info = self._fallback_model.get_model_info()  # type: ignore[union-attr]
        if self._online_err:
            info = {**info, "fallback_reason": self._online_err}
        return info


def create_smart_model(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> SmartTextModel:
    """Create a model that prefers online and falls back automatically."""
    return SmartTextModel(model=model, provider=provider, system_prompt=system_prompt)
