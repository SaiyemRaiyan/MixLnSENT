"""Per-model request settings.

These are measured, not guessed. Each entry records what a model actually needs
to return clean numbered output on this dataset, so the notebooks stay
model-agnostic: adding a model means adding one entry here.

Why the numbers differ:
  * Reasoning models spend part of the output budget on internal thinking before
    any answer is written. Give them too small a budget and the reply comes back
    empty with finish_reason="length" -- no error, just silence.
  * Groq additionally caps tokens per minute (8K for gpt-oss-120b), and the
    reserved output counts against it, so its batch size is bounded by
    input + reserved output < 8000.
"""

DEFAULT_SPEC = {
    "max_tokens": 2048,
    "batch_size": 50,
    "options": {},
}

MODEL_SPECS = {
    "groq": {
        "openai/gpt-oss-120b": {
            # batch is bounded by Groq's per-minute token cap, and the reserved output
            # counts against it. BnSentMix rows are ~20 tokens so 100 used to fit; on
            # SentMix-3L rows (~89 tokens) 50 is rejected with 413. Measured: 35 rows
            # is ~4.4k tokens and works for both datasets, so one setting covers both.
            "max_tokens": 4096,
            "batch_size": 35,
            "options": {"include_reasoning": False, "reasoning_effort": "low"},
        },
        "openai/gpt-oss-20b": {
            "max_tokens": 4096,
            "batch_size": 35,
            "options": {"include_reasoning": False, "reasoning_effort": "low"},
        },
        "openai/gpt-oss-safeguard-20b": {
            "max_tokens": 4096,
            "batch_size": 35,
            "options": {"include_reasoning": False, "reasoning_effort": "low"},
        },
        "qwen/qwen3.8-27b": {
            # reasoning is switched off entirely here, so a small output budget suffices.
            # measured: 35 rows works, 40 rows (~5.2k tokens) is rejected with 413
            "max_tokens": 80,
            "batch_size": 5,
            "options": {"reasoning_effort": "none", "reasoning_format": "hidden"},
        },
        "allam-2-7b": {
            # The tightest model in the lineup: a 4,096 token context and a 6,000
            # token per minute allowance. Both are measured, and both are the
            # binding constraint rather than any rate limit on requests.
            #
            # Measured token cost of this corpus for this model, from the API's own
            # accounting: a 1-row prompt is 779 tokens and a 3-row prompt is 2,222,
            # so these romanised-Bangla rows cost roughly 400-700 tokens each, not
            # the ~89 a naive character estimate suggests. With 256 reserved for
            # output that leaves under 3,840 for the batch, so only about six rows
            # fit. Batches of 10 and above are rejected with context_length_exceeded.
            #
            # The previous value of 25 was measured on BnSentMix, whose rows are far
            # shorter, and had to be corrected when it failed every batch here.
            "max_tokens": 256,
            "batch_size": 5,
            "options": {},
        },
    },
    "commandcode": {
        "meta/muse-spark-1.3-contributor": {
            # measured: reasoning + 400 answers needs ~8.7k output tokens
            "max_tokens": 16384,
            "batch_size": 400,
            "options": {"reasoning_effort": "low"},
        },
        "meta/muse-spark-1.3": {
            "max_tokens": 32768,
            "batch_size": 200,
            "options": {"reasoning_effort": "low"},
        },
        "google/gemini-3.8-flash": {
            # measured: 400/400 parsed, 2291 output tokens, no reasoning blow-up
            "max_tokens": 16384,
            "batch_size": 400,
            "options": {"reasoning_effort": "low"},
        },
        "google/gemini-3.7-flash": {
            "max_tokens": 16384,
            "batch_size": 400,
            "options": {"reasoning_effort": "low"},
        },
        "deepseek/deepseek-v4.1-flash": {
            # measured: 4096 truncates before any answer; 16384 parses 20/20
            "max_tokens": 16384,
            "batch_size": 100,
            "options": {"include_reasoning": False, "reasoning_effort": "low"},
        },
        # Claude models are only reachable through the Anthropic Messages shape
        # (/provider/v1/messages), not the OpenAI-compatible route. See providers.py.
        "claude-sonnet-4-6": {
            "max_tokens": 16384,
            "batch_size": 100,
            "options": {},
        },
        "gpt-5.6-sol": {
            # measured: cheapest frontier candidate, 161k tokens per test run
            "max_tokens": 16384,
            "batch_size": 100,
            "options": {"reasoning_effort": "low"},
        },
        "zai-org/GLM-5.3": {
            # measured: 174k tokens per run
            "max_tokens": 16384,
            "batch_size": 100,
            "options": {"reasoning_effort": "low"},
        },
        "Qwen/Qwen3.8-Max": {
            # measured: 300k tokens per run; scale test against Qwen3.8-27B
            "max_tokens": 16384,
            "batch_size": 100,
            "options": {"reasoning_effort": "low"},
        },
        "Qwen/Qwen3.8-27B": {
            # the same model as qwen/qwen3.8-27b on Groq, served here instead so the
            # Groq quota is not the binding constraint. measured: 35/35 parsed.
            # 262K context leaves the batch bounded by output, not input.
            "max_tokens": 16384,
            "batch_size": 200,
            "options": {"reasoning_effort": "low"},
        },
        "xai/grok-4.6": {
            "max_tokens": 16384,
            "batch_size": 100,
            "options": {"reasoning_effort": "low"},
        },
        "moonshotai/Kimi-K3": {
            "max_tokens": 16384,
            "batch_size": 100,
            "options": {"reasoning_effort": "low"},
        },
        "MiniMaxAI/MiniMax-M3": {
            "max_tokens": 16384,
            "batch_size": 100,
            "options": {"reasoning_effort": "low"},
        },
    },
    "gemini": {
        "gemini-3.6-flash": {"max_tokens": 8192, "batch_size": 50, "options": {}},
    },
    "openrouter": {
        # Entries are added per model once measured, as with every other provider.
        # OpenRouter routes to many upstreams, so batch size and output budget depend
        # on which one serves the request; the free variants are rate limited hard
        # enough that a modest batch is the safer default until measured.
        "qwen/qwen3.8-27b:free": {
            # OpenRouter's free tier caps requests per day, not tokens: 50 requests,
            # against 1,000 once 10 credits are on the account. Batch size is
            # therefore the thing that decides whether a run fits in a day, so it is
            # set as high as the model's 262k context and a 4096 output budget allow
            # rather than to the batch the same model uses on Groq.
            # 150 rows is ~13k input tokens and ~150 answer lines, so three bare
            # conditions on SentMix-3L cost 21 requests instead of 87.
            "max_tokens": 4096,
            "batch_size": 150,
            "options": {"reasoning_effort": "none", "reasoning_format": "hidden"},
        },
    },
    "huggingface": {
        "meta-llama/Llama-3.1-8B-Instruct": {
            # measured: 200/200 parsed at batch 200 in 19.1s
            "max_tokens": 2048,
            "batch_size": 200,
            "options": {},
        },
        "meta-llama/Llama-3.3-70B-Instruct": {
            # measured: 200/200 parsed at batch 200 in 11.2s
            "max_tokens": 2048,
            "batch_size": 200,
            "options": {},
        },
    },
    "ollama": {},
}


def model_spec(provider, model):
    """Return the measured request settings for a provider/model pair."""
    spec = MODEL_SPECS.get(provider, {}).get(model)
    if spec is None:
        return dict(DEFAULT_SPEC)
    merged = dict(DEFAULT_SPEC)
    merged.update(spec)
    return merged


def groq_options(model):
    """Backwards-compatible accessor for the Groq request options."""
    return model_spec("groq", model)["options"]
