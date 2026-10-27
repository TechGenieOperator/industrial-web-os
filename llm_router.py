"""
Industrial Web OS Multi-Provider LLM Router

Automatic failover order:
1. Groq
2. Cerebras Qwen
3. Cerebras GPT-OSS
4. Mistral
5. NVIDIA NIM
6. Google Gemini
7. Local Ollama

This router connects directly to providers.
It does not require the LiteLLM gateway or OpenRouter.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import urllib.error
import urllib.request
from crewai import BaseLLM
from dotenv import load_dotenv


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent
STATE_FILE = PROJECT_ROOT / "runs" / "llm_router_state.json"
EVENT_LOG_FILE = PROJECT_ROOT / "runs" / "llm_router_events.jsonl"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | LLM Router | %(message)s",
)

LOGGER = logging.getLogger("industrial_web_os.llm_router")


RETRYABLE_STATUS_CODES = {
    402,
    408,
    409,
    425,
    429,
    500,
    502,
    503,
    504,
    529,
}


PERMANENT_STATUS_CODES = {
    400,
    401,
    403,
    404,
    405,
    422,
}


RETRYABLE_ERROR_TEXT = (
    "rate limit",
    "rate_limit",
    "quota",
    "resource_exhausted",
    "resource exhausted",
    "insufficient credits",
    "budget exhausted",
    "in_flight_budget_exhausted",
    "too many requests",
    "temporarily unavailable",
    "service unavailable",
    "server overloaded",
    "overloaded",
    "connection error",
    "connection refused",
    "connection reset",
    "connection aborted",
    "timed out",
    "timeout",
    "gateway timeout",
    "internal server error",
    "try again later",
    "retry after",
)


PERMANENT_ERROR_TEXT = (
    "invalid api key",
    "incorrect api key",
    "unauthorized",
    "authentication failed",
    "permission denied",
    "forbidden",
    "model not found",
    "unsupported model",
    "invalid model",
    "access denied",
)


@dataclass
class Provider:
    name: str
    model: str
    endpoint: str
    api_key_env: Optional[str]
    context_window: int
    timeout_seconds: int = 180
    supports_tools: bool = True
    requires_api_key: bool = True
    enabled: bool = True

    def api_key(self) -> Optional[str]:
        if not self.api_key_env:
            return None

        value = os.getenv(self.api_key_env, "").strip()
        return value or None

    def configured(self) -> bool:
        if not self.enabled:
            return False

        if not self.model:
            return False

        if not self.endpoint:
            return False

        if self.requires_api_key and not self.api_key():
            return False

        return True

    def headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }

        key = self.api_key()

        if key:
            headers["Authorization"] = f"Bearer {key}"

        return headers


class ProviderError(RuntimeError):
    def __init__(
        self,
        provider_name: str,
        message: str,
        status_code: Optional[int] = None,
        retry_after_seconds: Optional[int] = None,
        retryable: bool = True,
        permanent_for_run: bool = False,
    ):
        super().__init__(message)

        self.provider_name = provider_name
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        self.retryable = retryable
        self.permanent_for_run = permanent_for_run


class AllProvidersFailedError(RuntimeError):
    pass


class MultiProviderLLMRouter(BaseLLM):
    def __init__(
        self,
        providers: Optional[List[Provider]] = None,
        temperature: float = 0.2,
        max_tokens: int = 16000,
        cooldown_seconds: Optional[int] = None,
        retries_per_provider: Optional[int] = None,
        max_tool_rounds: int = 30,
    ):
        super().__init__(
            model="industrial-multi-provider-router",
            temperature=temperature,
        )

        self.providers = providers or build_default_providers()
        self.max_tokens = max_tokens
        self.max_tool_rounds = max_tool_rounds

        self.cooldown_seconds = (
            cooldown_seconds
            if cooldown_seconds is not None
            else env_int("LLM_ROUTER_COOLDOWN_SECONDS", 3600)
        )

        self.retries_per_provider = (
            retries_per_provider
            if retries_per_provider is not None
            else env_int("LLM_ROUTER_MAX_RETRIES_PER_PROVIDER", 1)
        )

        self.disabled_for_run: Dict[str, str] = {}
        self.active_provider: Optional[Provider] = None
        self.state = self.load_state()

        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        EVENT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        configured_names = [
            provider.name
            for provider in self.providers
            if provider.configured()
        ]

        if configured_names:
            LOGGER.info(
                "Configured providers: %s",
                ", ".join(configured_names),
            )
        else:
            LOGGER.warning("No LLM providers are currently configured.")

    def call(
        self,
        messages: Union[str, List[Dict[str, Any]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> str:
        del callbacks

        normalized_messages = self.normalize_messages(messages)
        failures: List[Dict[str, Any]] = []

        for provider in self.providers:
            reason = self.unavailable_reason(provider)

            if reason:
                LOGGER.info(
                    "Skipping %s: %s",
                    provider.name,
                    reason,
                )

                self.log_event(
                    "provider_skipped",
                    provider,
                    reason=reason,
                )

                continue

            total_attempts = max(1, self.retries_per_provider + 1)

            for attempt in range(1, total_attempts + 1):
                try:
                    self.active_provider = provider

                    LOGGER.info(
                        "Trying %s with model %s, attempt %s/%s",
                        provider.name,
                        provider.model,
                        attempt,
                        total_attempts,
                    )

                    self.log_event(
                        "provider_attempt",
                        provider,
                        attempt=attempt,
                        total_attempts=total_attempts,
                    )

                    result = self.complete_tool_flow(
                        provider=provider,
                        messages=copy.deepcopy(normalized_messages),
                        tools=tools,
                        available_functions=available_functions,
                    )

                    self.clear_cooldown(provider.name)

                    LOGGER.info(
                        "Request completed using %s / %s",
                        provider.name,
                        provider.model,
                    )

                    self.log_event(
                        "provider_success",
                        provider,
                        attempt=attempt,
                    )

                    return result

                except ProviderError as error:
                    LOGGER.warning(
                        "%s failed: %s",
                        provider.name,
                        str(error),
                    )

                    failures.append(
                        {
                            "provider": provider.name,
                            "model": provider.model,
                            "status_code": error.status_code,
                            "message": str(error),
                            "retryable": error.retryable,
                            "attempt": attempt,
                        }
                    )

                    self.log_event(
                        "provider_failure",
                        provider,
                        status_code=error.status_code,
                        message=str(error),
                        retryable=error.retryable,
                        attempt=attempt,
                    )

                    if error.permanent_for_run or not error.retryable:
                        self.disabled_for_run[provider.name] = str(error)
                        break

                    if attempt >= total_attempts:
                        cooldown = (
                            error.retry_after_seconds
                            or self.cooldown_seconds
                        )

                        self.set_cooldown(
                            provider.name,
                            cooldown,
                            str(error),
                        )

                        break

                    wait_seconds = min(
                        error.retry_after_seconds or (2 ** attempt),
                        30,
                    )

                    LOGGER.info(
                        "Waiting %s seconds before retrying %s",
                        wait_seconds,
                        provider.name,
                    )

                    time.sleep(wait_seconds)

                except Exception as error:
                    LOGGER.exception(
                        "Unexpected failure from %s",
                        provider.name,
                    )

                    failures.append(
                        {
                            "provider": provider.name,
                            "model": provider.model,
                            "status_code": None,
                            "message": str(error),
                            "retryable": True,
                            "attempt": attempt,
                        }
                    )

                    self.set_cooldown(
                        provider.name,
                        self.cooldown_seconds,
                        str(error),
                    )

                    break

        self.active_provider = None

        raise AllProvidersFailedError(
            "All configured providers failed, were unavailable, "
            "or were on cooldown.\n\n"
            + json.dumps(
                failures,
                indent=2,
                ensure_ascii=False,
            )
        )

    def complete_tool_flow(
        self,
        provider: Provider,
        messages: List[Dict[str, Any]],
        tools: Optional[List[dict]],
        available_functions: Optional[Dict[str, Any]],
    ) -> str:
        for tool_round in range(1, self.max_tool_rounds + 1):
            response_data = self.request_completion(
                provider=provider,
                messages=messages,
                tools=tools,
            )

            try:
                message = response_data["choices"][0]["message"]
            except (KeyError, IndexError, TypeError) as error:
                raise ProviderError(
                    provider.name,
                    f"Invalid response structure: {error}",
                    retryable=True,
                ) from error

            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                content = message.get("content")

                if content is None:
                    raise ProviderError(
                        provider.name,
                        "Provider returned no content and no tool calls.",
                        retryable=True,
                    )

                return str(content)

            if not available_functions:
                raise ProviderError(
                    provider.name,
                    "The provider requested a tool, but CrewAI did not "
                    "provide available functions.",
                    retryable=False,
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": tool_calls,
                }
            )

            for tool_call in tool_calls:
                tool_result = self.execute_tool_call(
                    provider=provider,
                    tool_call=tool_call,
                    available_functions=available_functions,
                )

                messages.append(tool_result)

            self.log_event(
                "tool_round_completed",
                provider,
                tool_round=tool_round,
                tool_count=len(tool_calls),
            )

        raise ProviderError(
            provider.name,
            f"Maximum tool rounds exceeded: {self.max_tool_rounds}",
            retryable=False,
        )

    def execute_tool_call(
        self,
        provider: Provider,
        tool_call: Dict[str, Any],
        available_functions: Dict[str, Any],
    ) -> Dict[str, Any]:
        tool_call_id = tool_call.get("id", "unknown")

        try:
            function_data = tool_call["function"]
            function_name = function_data["name"]
            raw_arguments = function_data.get("arguments") or "{}"
            arguments = json.loads(raw_arguments)

        except (KeyError, TypeError, json.JSONDecodeError) as error:
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(
                    {
                        "error": "Invalid tool call",
                        "details": str(error),
                    }
                ),
            }

        function = available_functions.get(function_name)

        if function is None:
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": function_name,
                "content": json.dumps(
                    {
                        "error": f"Tool '{function_name}' is unavailable."
                    }
                ),
            }

        try:
            result = function(**arguments)

            if result is None:
                result = ""

            if isinstance(result, str):
                content = result
            else:
                content = json.dumps(
                    result,
                    ensure_ascii=False,
                    default=str,
                )

            self.log_event(
                "tool_executed",
                provider,
                tool_name=function_name,
                success=True,
            )

            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": function_name,
                "content": content,
            }

        except Exception as error:
            LOGGER.exception(
                "Tool %s failed while using %s",
                function_name,
                provider.name,
            )

            self.log_event(
                "tool_executed",
                provider,
                tool_name=function_name,
                success=False,
                error=str(error),
            )

            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": function_name,
                "content": json.dumps(
                    {
                        "error": f"Tool '{function_name}' failed.",
                        "details": str(error),
                    },
                    ensure_ascii=False,
                ),
            }

    def request_completion(
        self,
        provider: Provider,
        messages: List[Dict[str, Any]],
        tools: Optional[List[dict]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": provider.model,
            "messages": messages,
            "stream": False,
        }

        if self.temperature is not None:
            payload["temperature"] = self.temperature

        if self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens

        if tools and provider.supports_tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            request = urllib.request.Request(
                provider.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=provider.headers(),
                method="POST",
            )
            try:
                response = urllib.request.urlopen(
                    request,
                    timeout=provider.timeout_seconds,
                )
                status_code = response.status
                response_text = response.read().decode("utf-8", errors="replace")
                response_headers = response.headers
            except urllib.error.HTTPError as error:
                status_code = error.code
                response_text = error.read().decode("utf-8", errors="replace")
                response_headers = error.headers

        except TimeoutError as error:
            raise ProviderError(
                provider.name,
                f"Request timed out after "
                f"{provider.timeout_seconds} seconds.",
                retryable=True,
            ) from error

        except urllib.error.URLError as error:
            raise ProviderError(
                provider.name,
                f"Connection failed: {error}",
                retryable=True,
            ) from error

        except OSError as error:
            raise ProviderError(
                provider.name,
                f"Request failed: {error}",
                retryable=True,
            ) from error

        if 200 <= status_code < 300:
            try:
                return json.loads(response_text)

            except ValueError as error:
                raise ProviderError(
                    provider.name,
                    "Provider returned a successful response "
                    "that was not valid JSON.",
                    status_code=status_code,
                    retryable=True,
                ) from error

        retry_after = self.extract_retry_after(response_headers, response_text)

        retryable, permanent = self.classify_failure(
            status_code=status_code,
            message=response_text,
        )

        raise ProviderError(
            provider.name,
            f"HTTP {status_code}: {response_text[:2000]}",
            status_code=status_code,
            retry_after_seconds=retry_after,
            retryable=retryable,
            permanent_for_run=permanent,
        )

    @staticmethod
    def classify_failure(
        status_code: Optional[int],
        message: str,
    ) -> tuple[bool, bool]:
        normalized = message.lower()

        if any(
            text in normalized
            for text in PERMANENT_ERROR_TEXT
        ):
            return False, True

        if any(
            text in normalized
            for text in RETRYABLE_ERROR_TEXT
        ):
            return True, False

        if status_code in RETRYABLE_STATUS_CODES:
            return True, False

        if status_code in PERMANENT_STATUS_CODES:
            return False, True

        return True, False

    @staticmethod
    def extract_retry_after(
        headers: Any,
        body_text: str,
    ) -> Optional[int]:
        retry_after_header = headers.get("Retry-After")

        if retry_after_header:
            try:
                return max(
                    1,
                    int(float(retry_after_header)),
                )
            except ValueError:
                pass

        patterns = (
            r'"retryDelay"\s*:\s*"(\d+)s"',
            r"retry in\s+(\d+)s",
            r"retry after\s+(\d+)",
        )

        for pattern in patterns:
            match = re.search(
                pattern,
                body_text,
                flags=re.IGNORECASE,
            )

            if match:
                return max(
                    1,
                    int(match.group(1)),
                )

        return None

    @staticmethod
    def normalize_messages(
        messages: Union[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if isinstance(messages, str):
            return [
                {
                    "role": "user",
                    "content": messages,
                }
            ]

        return copy.deepcopy(messages)

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        windows = [
            provider.context_window
            for provider in self.providers
            if provider.configured()
        ]

        if not windows:
            return 32000

        return min(windows)

    def unavailable_reason(
        self,
        provider: Provider,
    ) -> Optional[str]:
        if not provider.enabled:
            return "provider disabled"

        if not provider.model:
            return "model not configured"

        if not provider.endpoint:
            return "endpoint not configured"

        if provider.requires_api_key and not provider.api_key():
            return f"missing {provider.api_key_env}"

        if provider.name in self.disabled_for_run:
            return (
                "disabled for this run: "
                + self.disabled_for_run[provider.name]
            )

        cooldown = self.state.get(
            "cooldowns",
            {},
        ).get(
            provider.name,
            {},
        )

        cooldown_until = float(
            cooldown.get("until", 0) or 0
        )

        if cooldown_until > time.time():
            remaining = int(
                cooldown_until - time.time()
            )

            return (
                f"cooldown active for "
                f"{remaining} more seconds"
            )

        if cooldown:
            self.clear_cooldown(provider.name)

        return None

    def set_cooldown(
        self,
        provider_name: str,
        seconds: int,
        reason: str,
    ) -> None:
        safe_seconds = max(
            1,
            int(seconds),
        )

        self.state.setdefault(
            "cooldowns",
            {},
        )[provider_name] = {
            "until": time.time() + safe_seconds,
            "seconds": safe_seconds,
            "reason": reason,
            "created_at": time.time(),
        }

        self.save_state()

        LOGGER.warning(
            "%s placed on cooldown for %s seconds",
            provider_name,
            safe_seconds,
        )

    def clear_cooldown(
        self,
        provider_name: str,
    ) -> None:
        cooldowns = self.state.setdefault(
            "cooldowns",
            {},
        )

        if provider_name in cooldowns:
            del cooldowns[provider_name]
            self.save_state()

    def load_state(self) -> Dict[str, Any]:
        if not STATE_FILE.exists():
            return {
                "cooldowns": {},
            }

        try:
            with STATE_FILE.open(
                "r",
                encoding="utf-8",
            ) as file:
                state = json.load(file)

            if not isinstance(state, dict):
                raise ValueError(
                    "Router state must be a JSON object."
                )

            state.setdefault(
                "cooldowns",
                {},
            )

            return state

        except Exception as error:
            LOGGER.warning(
                "Could not load router state: %s",
                error,
            )

            return {
                "cooldowns": {},
            }

    def save_state(self) -> None:
        STATE_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_file = STATE_FILE.with_suffix(
            ".tmp"
        )

        with temporary_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.state,
                file,
                indent=2,
                ensure_ascii=False,
            )

        temporary_file.replace(STATE_FILE)

    def reset_cooldowns(self) -> None:
        self.state["cooldowns"] = {}
        self.save_state()

        LOGGER.info(
            "All provider cooldowns were reset."
        )

    def provider_status(self) -> List[Dict[str, Any]]:
        current_time = time.time()
        statuses: List[Dict[str, Any]] = []

        for provider in self.providers:
            cooldown = self.state.get(
                "cooldowns",
                {},
            ).get(
                provider.name,
                {},
            )

            cooldown_until = float(
                cooldown.get("until", 0) or 0
            )

            statuses.append(
                {
                    "name": provider.name,
                    "model": provider.model,
                    "configured": provider.configured(),
                    "disabled_for_run": (
                        provider.name
                        in self.disabled_for_run
                    ),
                    "cooldown_active": (
                        cooldown_until > current_time
                    ),
                    "cooldown_remaining_seconds": max(
                        0,
                        int(
                            cooldown_until
                            - current_time
                        ),
                    ),
                    "last_failure": cooldown.get(
                        "reason"
                    ),
                }
            )

        return statuses

    def log_event(
        self,
        event_type: str,
        provider: Provider,
        **details: Any,
    ) -> None:
        event = {
            "timestamp": time.time(),
            "event": event_type,
            "provider": provider.name,
            "model": provider.model,
            **details,
        }

        try:
            EVENT_LOG_FILE.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with EVENT_LOG_FILE.open(
                "a",
                encoding="utf-8",
            ) as file:
                file.write(
                    json.dumps(
                        event,
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )

        except Exception as error:
            LOGGER.warning(
                "Could not write router log: %s",
                error,
            )


def build_default_providers() -> List[Provider]:
    ollama_enabled = env_bool(
        "LLM_ROUTER_ENABLE_LOCAL_FALLBACK",
        True,
    )

    return [
        Provider(
            name="groq-primary",
            model=os.getenv(
                "GROQ_MODEL",
                "",
            ).strip(),
            endpoint=os.getenv(
                "GROQ_CHAT_ENDPOINT",
                "https://api.groq.com/openai/v1/chat/completions",
            ).strip(),
            api_key_env="GROQ_API_KEY",
            context_window=env_int(
                "GROQ_CONTEXT_WINDOW",
                128000,
            ),
            timeout_seconds=env_int(
                "GROQ_TIMEOUT_SECONDS",
                180,
            ),
        ),
        Provider(
            name="cerebras-qwen",
            model=os.getenv(
                "CEREBRAS_PRIMARY_MODEL",
                "qwen-3.8-27b",
            ).strip(),
            endpoint=os.getenv(
                "CEREBRAS_CHAT_ENDPOINT",
                "https://api.cerebras.ai/v1/chat/completions",
            ).strip(),
            api_key_env="CEREBRAS_API_KEY",
            context_window=env_int(
                "CEREBRAS_PRIMARY_CONTEXT_WINDOW",
                64000,
            ),
            timeout_seconds=env_int(
                "CEREBRAS_TIMEOUT_SECONDS",
                180,
            ),
        ),
        Provider(
            name="cerebras-gpt-oss",
            model=os.getenv(
                "CEREBRAS_FALLBACK_MODEL",
                "gpt-oss-120b",
            ).strip(),
            endpoint=os.getenv(
                "CEREBRAS_CHAT_ENDPOINT",
                "https://api.cerebras.ai/v1/chat/completions",
            ).strip(),
            api_key_env="CEREBRAS_API_KEY",
            context_window=env_int(
                "CEREBRAS_FALLBACK_CONTEXT_WINDOW",
                65000,
            ),
            timeout_seconds=env_int(
                "CEREBRAS_TIMEOUT_SECONDS",
                180,
            ),
        ),
        Provider(
            name="mistral",
            model=os.getenv(
                "MISTRAL_MODEL",
                "",
            ).strip(),
            endpoint=os.getenv(
                "MISTRAL_CHAT_ENDPOINT",
                "https://api.mistral.ai/v1/chat/completions",
            ).strip(),
            api_key_env="MISTRAL_API_KEY",
            context_window=env_int(
                "MISTRAL_CONTEXT_WINDOW",
                128000,
            ),
            timeout_seconds=env_int(
                "MISTRAL_TIMEOUT_SECONDS",
                180,
            ),
        ),
        Provider(
            name="nvidia-nim",
            model=os.getenv(
                "NVIDIA_MODEL",
                "",
            ).strip(),
            endpoint=os.getenv(
                "NVIDIA_CHAT_ENDPOINT",
                "https://integrate.api.nvidia.com/v1/chat/completions",
            ).strip(),
            api_key_env="NVIDIA_API_KEY",
            context_window=env_int(
                "NVIDIA_CONTEXT_WINDOW",
                128000,
            ),
            timeout_seconds=env_int(
                "NVIDIA_TIMEOUT_SECONDS",
                240,
            ),
        ),
        Provider(
            name="google-gemini",
            model=os.getenv(
                "GEMINI_MODEL",
                "gemini-3.8-flash",
            ).strip(),
            endpoint=os.getenv(
                "GEMINI_CHAT_ENDPOINT",
                (
                    "https://generativelanguage.googleapis.com/"
                    "v1beta/openai/chat/completions"
                ),
            ).strip(),
            api_key_env="GEMINI_API_KEY",
            context_window=env_int(
                "GEMINI_CONTEXT_WINDOW",
                1048576,
            ),
            timeout_seconds=env_int(
                "GEMINI_TIMEOUT_SECONDS",
                240,
            ),
        ),
        Provider(
            name="ollama-local",
            model=os.getenv(
                "OLLAMA_MODEL",
                "qwen3:1.7b",
            ).strip(),
            endpoint=join_endpoint(
                os.getenv(
                    "OLLAMA_BASE_URL",
                    "http://127.0.0.1:11434",
                ),
                "/v1/chat/completions",
            ),
            api_key_env=None,
            context_window=env_int(
                "OLLAMA_CONTEXT_WINDOW",
                16384,
            ),
            timeout_seconds=env_int(
                "OLLAMA_TIMEOUT_SECONDS",
                900,
            ),
            requires_api_key=False,
            enabled=ollama_enabled,
        ),
    ]


def create_router() -> MultiProviderLLMRouter:
    return MultiProviderLLMRouter(
        temperature=env_float(
            "LLM_ROUTER_TEMPERATURE",
            0.2,
        ),
        max_tokens=env_int(
            "LLM_ROUTER_MAX_TOKENS",
            16000,
        ),
        max_tool_rounds=env_int(
            "LLM_ROUTER_MAX_TOOL_ROUNDS",
            30,
        ),
    )


def join_endpoint(
    base_url: str,
    path: str,
) -> str:
    normalized_base = base_url.strip().rstrip("/")
    normalized_path = "/" + path.strip().lstrip("/")

    return normalized_base + normalized_path


def env_int(
    name: str,
    default: int,
) -> int:
    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    try:
        return int(value)

    except ValueError:
        LOGGER.warning(
            "Invalid integer for %s. Using %s.",
            name,
            default,
        )

        return default


def env_float(
    name: str,
    default: float,
) -> float:
    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    try:
        return float(value)

    except ValueError:
        LOGGER.warning(
            "Invalid number for %s. Using %s.",
            name,
            default,
        )

        return default


def env_bool(
    name: str,
    default: bool,
) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    normalized = value.strip().lower()

    if normalized in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True

    if normalized in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False

    return default


if __name__ == "__main__":
    router = create_router()

    print(
        json.dumps(
            router.provider_status(),
            indent=2,
            ensure_ascii=False,
        )
    )