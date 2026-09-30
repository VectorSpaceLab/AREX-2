import asyncio
from typing import Any, Callable, Optional

from openai import AsyncOpenAI


def is_rate_limit_error(exc: Exception) -> bool:
    error_type = type(exc).__name__.lower()
    error_msg = str(exc).lower()
    return (
        "ratelimit" in error_type
        or "rate limit" in error_msg
        or "ratelimit" in error_msg
        or "too many requests" in error_msg
        or "429" in error_msg
    )


class _AsyncRetryingCompletions:
    def __init__(
        self,
        completions: Any,
        *,
        general_max_attempts: int,
        rate_limit_max_retries: int,
        rate_limit_sleep: float,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self._completions = completions
        self._general_max_attempts = general_max_attempts
        self._rate_limit_max_retries = rate_limit_max_retries
        self._rate_limit_sleep = rate_limit_sleep
        self._logger = logger

    def _log(self, message: str):
        if self._logger is not None:
            self._logger(message)

    async def create(self, *args, **kwargs):
        general_attempt = 0
        rate_limit_retry_count = 0
        while True:
            try:
                # return await self._completions.create(*args, **kwargs)
                response = await self._completions.create(*args, **kwargs)
                content = response.choices[0].message.content or ""
                if '<tool_call>' in content and '</tool_call>' not in content:
                    response.choices[0].message.content = response.choices[0].message.content.strip() + '\n</tool_call>'
                content = response.choices[0].message.content
                # print("--------------------------------")
                # print("--------------------------------")
                # print("--------------------------------")
                # print(content)
                # print("--------------------------------")
                # print("--------------------------------")
                # print("--------------------------------")
                return response
            except Exception as exc:
                if is_rate_limit_error(exc):
                    rate_limit_retry_count += 1
                    if rate_limit_retry_count > self._rate_limit_max_retries:
                        raise
                    self._log(
                        f"[WARN] OpenAI async rate limit retry "
                        f"{rate_limit_retry_count}/{self._rate_limit_max_retries}: {exc}"
                    )
                    await asyncio.sleep(self._rate_limit_sleep)
                    continue

                general_attempt += 1
                if general_attempt >= self._general_max_attempts:
                    raise
                self._log(
                    f"[WARN] OpenAI async request retry "
                    f"{general_attempt}/{self._general_max_attempts}: {exc}"
                )


class _ChatWrapper:
    def __init__(self, chat: Any, completions_wrapper: Any):
        self._chat = chat
        self.completions = completions_wrapper

    def __getattr__(self, name: str):
        return getattr(self._chat, name)


class RetryingAsyncOpenAI:
    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        general_max_attempts: int = 5,
        rate_limit_max_retries: int = 60,
        rate_limit_sleep: float = 3.0,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self._client = client
        self.chat = _ChatWrapper(
            client.chat,
            _AsyncRetryingCompletions(
                client.chat.completions,
                general_max_attempts=general_max_attempts,
                rate_limit_max_retries=rate_limit_max_retries,
                rate_limit_sleep=rate_limit_sleep,
                logger=logger,
            ),
        )
    def __getattr__(self, name: str):
        return getattr(self._client, name)


def create_async_openai_with_retry(
    *,
    general_max_attempts: int = 5,
    rate_limit_max_retries: int = 60,
    rate_limit_sleep: float = 3.0,
    logger: Optional[Callable[[str], None]] = None,
    **client_kwargs,
):
    client = AsyncOpenAI(**client_kwargs)
    return RetryingAsyncOpenAI(
        client,
        general_max_attempts=general_max_attempts,
        rate_limit_max_retries=rate_limit_max_retries,
        rate_limit_sleep=rate_limit_sleep,
        logger=logger,
    )
