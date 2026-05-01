from __future__ import annotations

import json
from typing import Any, Protocol

from support_agent.core.config import Settings


class LLMProvider(Protocol):
    name: str

    def complete_json(self, system: str, user: str) -> dict[str, Any] | None:
        ...


class TemplateProvider:
    name = "template"

    def complete_json(self, system: str, user: str) -> dict[str, Any] | None:
        return None


class GroqProvider:
    name = "groq"

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def complete_json(self, system: str, user: str) -> dict[str, Any] | None:
        if not self.api_key:
            return None
        try:
            from groq import Groq

            client = Groq(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception:
            return None


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def complete_json(self, system: str, user: str) -> dict[str, Any] | None:
        if not self.api_key:
            return None
        try:
            from google import genai

            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=f"{system}\n\n{user}",
                config={"temperature": 0, "response_mime_type": "application/json"},
            )
            return json.loads(response.text or "{}")
        except Exception:
            return None


class DockerModelRunnerProvider:
    name = "docker_model_runner"

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url
        self.model = model

    def complete_json(self, system: str, user: str) -> dict[str, Any] | None:
        try:
            from openai import OpenAI

            client = OpenAI(base_url=self.base_url, api_key="docker-model-runner")
            response = client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception:
            return None


class LLMRouter:
    def __init__(self, settings: Settings) -> None:
        provider = settings.llm_provider.lower()
        providers: list[LLMProvider] = []

        if provider in {"auto", "groq"}:
            providers.append(GroqProvider(settings.groq_api_key, settings.groq_model))
        if provider in {"auto", "gemini"}:
            providers.append(GeminiProvider(settings.gemini_api_key, settings.gemini_model))
        if provider in {"auto", "dmr", "docker_model_runner", "local"}:
            providers.append(DockerModelRunnerProvider(settings.dmr_base_url, settings.dmr_model))
        providers.append(TemplateProvider())
        self.providers = providers

    def complete_json(self, system: str, user: str) -> tuple[dict[str, Any] | None, str]:
        for provider in self.providers:
            result = provider.complete_json(system, user)
            if result is not None:
                return result, provider.name
        return None, "none"
