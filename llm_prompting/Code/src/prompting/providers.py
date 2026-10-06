"""Provider adapters for the decoupled prompting experiment.

Provider-specific request shape lives here; per-model budgets and options live in
models.py. Notebooks call `build_sender(provider, model)` and never import a
provider SDK, so switching provider or model means changing one variable.

Several keys per provider are supported. Set GROQ_API_KEY_1, GROQ_API_KEY_2, ...
and the sender rotates to the next key on its own when one runs out of quota, so a
long run continues instead of stopping. A single unsuffixed key (GROQ_API_KEY)
also works, and is tried first.
"""

import os
import re
import time

from .models import model_spec

KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "huggingface": "HF_TOKEN",
    "commandcode": "COMMANDCODE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
MAX_KEYS = 10

RATE_LIMIT_MARKERS = (
    "rate limit", "rate_limit", "ratelimit", "resource_exhausted",
    "resource exhausted", "quota", "too many requests", "429",
    # an account that has run out of credits or allowance cannot serve more
    # requests either, so rotation applies to these too
    "payment required", "depleted", "insufficient", "exceeded your current",
)
QUOTA_STATUS_CODES = (402, 429)


# An aggregator can fail a request because the upstream pool serving that model is
# busy, which says nothing about our own key or allowance. Retrying the same key is
# correct there, whereas rotating wastes a working key and stopping ends the run.
# OpenRouter reports this as a 429 with these markers.
UPSTREAM_CONTENTION_MARKERS = (
    "temporarily rate-limited upstream",
    "shared_pool",
    "provider returned error",
)


def is_upstream_contention(error):
    """True when the upstream serving the model is busy, not our key's quota.

    Distinguishing the two matters: this is transient and the same key will work in
    a moment, so it must be retried rather than treated as an exhausted account.
    """
    text = f"{type(error).__name__} {error}".lower()
    return any(marker in text for marker in UPSTREAM_CONTENTION_MARKERS)


def is_request_too_large(error):
    """True when the provider rejected the request for exceeding a size limit.

    A batch the provider refuses can be too large for two different reasons: it
    exceeds the model's context window outright, or it exceeds the per-minute token
    allowance. Either way the remedy is to split the batch rather than rotate keys,
    because no key change makes an oversized request fit.

    The test is the status code, not the wording. Groq says "on tokens per minute
    (TPM)" in both the 413 that means the request is too large and the 429 that
    means the minute's allowance is spent, so matching that phrase conflates a
    request the provider will never accept with one it will accept shortly.
    """
    status = (
        getattr(error, "status_code", None)
        or getattr(error, "code", None)
        or getattr(getattr(error, "response", None), "status_code", None)
    )
    if status == 413:
        return True
    if status == 400:
        text = f"{type(error).__name__} {error}".lower()
        return any(
            marker in text
            for marker in (
                "request too large",
                "context_length_exceeded",
                "reduce the length",
                "context length",
                "maximum context",
            )
        )
    return False


RETRY_HINT = re.compile(r"try again in ([\d.]+)\s*s", re.IGNORECASE)


def retry_after_seconds(error):
    """Seconds to wait before retrying, when the provider states one.

    A per-minute allowance resets on a clock the provider names in the message, so
    waiting exactly that long is both sufficient and the cheapest response.
    """
    match = RETRY_HINT.search(str(error))
    return float(match.group(1)) if match else None


def is_rate_limit_error(error):
    """True when the current key/account can no longer serve requests.

    Covers rate limits and exhausted credits, because both mean the same thing to
    a run: try the next key, and only stop when every key has been used.
    """
    if is_upstream_contention(error) or is_request_too_large(error):
        return False
    status = (
        getattr(error, "status_code", None)
        or getattr(error, "code", None)
        or getattr(getattr(error, "response", None), "status_code", None)
    )
    if status in QUOTA_STATUS_CODES:
        return True
    text = f"{type(error).__name__} {error}".lower()
    return any(marker in text for marker in RATE_LIMIT_MARKERS)


def provider_keys(provider):
    """Ordered keys for a provider: the plain name first, then _1, _2, ..."""
    base = KEY_ENV.get(provider)
    if base is None:
        return [None]
    names = [base] + [f"{base}_{number}" for number in range(1, MAX_KEYS)]
    return [os.environ[name] for name in names if os.environ.get(name)]


def _groq_client(key):
    from groq import Groq
    return Groq(api_key=key)


def _gemini_client(key):
    from google import genai
    return genai.Client(api_key=key)


def _huggingface_client(key):
    from huggingface_hub import InferenceClient
    return InferenceClient(api_key=key)


def _commandcode_client(key):
    from openai import OpenAI
    return OpenAI(base_url="https://api.commandcode.ai/provider/v1", api_key=key)


def _openrouter_client(key):
    from openai import OpenAI
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def _ollama_client(key):
    from openai import OpenAI
    return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def _groq_send(client, model, prompt):
    spec = model_spec("groq", model)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_completion_tokens=spec["max_tokens"],
        **spec["options"],
    )
    return response.choices[0].message.content


def _gemini_send(client, model, prompt):
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"temperature": 0},
    )
    return response.text


def _huggingface_send(client, model, prompt):
    spec = model_spec("huggingface", model)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=spec["max_tokens"],
        **spec["options"],
    )
    return response.choices[0].message.content


# parameters the OpenAI SDK accepts as real keyword arguments; anything else a
# provider wants must travel in the request body via extra_body, otherwise the SDK
# raises "unexpected keyword argument"
NATIVE_OPENAI_OPTIONS = {
    "reasoning_effort", "top_p", "seed", "stop",
    "frequency_penalty", "presence_penalty", "logprobs", "top_logprobs",
}


def _split_options(options):
    native = {key: value for key, value in options.items() if key in NATIVE_OPENAI_OPTIONS}
    extra = {key: value for key, value in options.items() if key not in NATIVE_OPENAI_OPTIONS}
    return native, extra


def _commandcode_send(client, model, prompt):
    spec = model_spec("commandcode", model)
    native, extra = _split_options(spec["options"])
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=spec["max_tokens"],
        extra_body=extra or None,
        **native,
    )
    return response.choices[0].message.content


def _openrouter_send(client, model, prompt):
    spec = model_spec("openrouter", model)
    native, extra = _split_options(spec["options"])
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=spec["max_tokens"],
        extra_body=extra or None,
        **native,
    )
    return response.choices[0].message.content


ANTHROPIC_MESSAGES_URL = "https://api.commandcode.ai/provider/v1/messages"
ANTHROPIC_PREFIXES = ("claude-",)


def needs_anthropic_shape(model):
    """Claude models are only served through the Anthropic Messages shape.

    The provider rejects them on the OpenAI-compatible route with
    'must be called via /provider/v1/messages'.
    """
    return model.startswith(ANTHROPIC_PREFIXES)


def _commandcode_anthropic_send(key, model, prompt):
    import httpx

    spec = model_spec("commandcode", model)
    response = httpx.post(
        ANTHROPIC_MESSAGES_URL,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": spec["max_tokens"],
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
            **spec["options"],
        },
        timeout=600,
    )
    response.raise_for_status()
    payload = response.json()
    return "".join(
        block.get("text", "")
        for block in payload.get("content", [])
        if block.get("type") == "text"
    )


def _ollama_send(client, model, prompt):
    spec = model_spec("ollama", model)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=spec["max_tokens"],
        **spec["options"],
    )
    return response.choices[0].message.content


CLIENTS = {
    "groq": _groq_client,
    "gemini": _gemini_client,
    "huggingface": _huggingface_client,
    "commandcode": _commandcode_client,
    "openrouter": _openrouter_client,
    "ollama": _ollama_client,
}
SENDERS = {
    "groq": _groq_send,
    "gemini": _gemini_send,
    "huggingface": _huggingface_send,
    "commandcode": _commandcode_send,
    "openrouter": _openrouter_send,
    "ollama": _ollama_send,
}


def build_sender(provider, model, log=print):
    """Return send(prompt) for one provider/model, rotating keys on quota errors.

    Clients are created lazily, one per key, and kept for the life of the run. A
    rate-limited request is retried on the next key; only when every key has been
    tried does the original error propagate, which the runner turns into a clean
    stop so the run can resume later.
    """
    if provider not in SENDERS:
        raise ValueError(f"Unknown provider {provider!r}. Known: {', '.join(SENDERS)}")

    keys = provider_keys(provider)
    if provider in KEY_ENV and not keys:
        raise RuntimeError(f"{KEY_ENV[provider]} is not set. Add it to .env")

    clients = {}
    state = {"index": 0}

    def client_at(index):
        if index not in clients:
            clients[index] = CLIENTS[provider](keys[index])
        return clients[index]

    def send(prompt):
        while True:
            index = state["index"]
            try:
                if provider == "commandcode" and needs_anthropic_shape(model):
                    return _commandcode_anthropic_send(keys[index], model, prompt)
                return SENDERS[provider](client_at(index), model, prompt)
            except Exception as error:
                if not is_rate_limit_error(error) or index + 1 >= len(keys):
                    raise
                # Rotate first, even when the provider suggests a wait. Keys on the
                # free tier sit in separate organisations, which the errors name, so
                # each carries its own allowance and moving to the next one resumes
                # work immediately. Measured on allam-2-7b, rotating reaches roughly
                # twice the row rate of waiting on a single key. Only once every key
                # has been tried does the wait become the right move, and the runner
                # handles that using the delay the provider reports.
                state["index"] = index + 1
                log(
                    f"  key {index + 1}/{len(keys)} out of quota, "
                    f"switching to key {index + 2}/{len(keys)}"
                )

    send.key_count = len(keys)
    send.current_key = lambda: state["index"] + 1
    return send
