"""The model call, its providers, prices and the daily budget — lifted out of webapp/app.py
so the evaluation driver and the benchmark CLI can run without FastAPI.

`chat(...)` is the one place a Claude/DeepSeek/Bedrock/Foundry request is built. Sampling controls
(`temperature`, `thinking`, `effort`) are opt-in keyword arguments: when None (the product
default) nothing is sent and the provider's defaults apply, exactly as before. The benchmark
pins them and stamps them on the run. `supports_sampling` refuses `temperature` on models
that reject it (Sonnet 5 / Opus 5+ / Fable return HTTP 400), so a pinned run fails fast
instead of burning a run's worth of 400s.

Azure Foundry is two providers: `foundry` (Claude, Anthropic SDK) and `foundry_openai`
(OpenAI models such as gpt-5.4, OpenAI SDK against the Foundry `/openai/v1/` route).
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from . import ai_models, peak_schedule, pricing, settings
from .backend import db

_IS_PG = db.__name__.endswith("db_pg")
_P = "%s" if _IS_PG else "?"

# Provider is chosen in the UI (Settings page) and stored in app_settings. API keys still
# come from env; the toggle just picks which provider to use.
PROVIDERS = {
    "anthropic": {"label": "Anthropic (Claude)", "base_url": "",
                  "model": "claude-haiku-4-5-20251001",
                  "key_envs": ("ANTHROPIC_API_KEY", "AI_API_KEY"),
                  "price_in": 1.0, "price_out": 5.0},
    "deepseek": {"label": "DeepSeek", "base_url": "https://api.deepseek.com/anthropic",
                 "model": "deepseek-chat",
                 "key_envs": ("DEEPSEEK_API_KEY", "AI_API_KEY"),
                 "price_in": 0.27, "price_out": 1.10},
    # AWS Bedrock serves Claude models through the same Anthropic SDK (AnthropicBedrock
    # client). It authenticates with AWS credentials rather than a single API key — see
    # bedrock_creds — and takes Bedrock model IDs / cross-region inference-profile IDs,
    # which are entered per model on the Settings page. The default below is only a
    # placeholder fallback; add the real IDs (e.g. eu.anthropic.claude-...:0) in Settings.
    "bedrock": {"label": "AWS Bedrock (Claude)", "base_url": "",
                "model": "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
                "key_envs": (),
                "price_in": 1.0, "price_out": 5.0},
    # Azure AI Foundry serves Claude models through the same Anthropic SDK
    # (AnthropicFoundry client) and the same Messages API — see foundry_creds.
    # `model` is the Foundry *deployment name* (it defaults to the model id, e.g.
    # claude-haiku-4-5, but can be customised in the Foundry portal), entered per
    # model on the Settings page.
    "foundry": {"label": "Azure Foundry (Claude)", "base_url": "",
                "model": "claude-haiku-4-5",
                "key_envs": ("ANTHROPIC_FOUNDRY_API_KEY",),
                "price_in": 1.0, "price_out": 5.0},
    # Same Foundry resource, OpenAI v1 route. `model` is the deployment name (it
    # defaults to the model id, e.g. gpt-5.4). A key/endpoint set for Claude is
    # reused when the OpenAI-specific vars are absent — see foundry_openai_creds.
    "foundry_openai": {"label": "Azure Foundry (OpenAI)", "base_url": "",
                       "model": "gpt-5.4-mini",
                       "key_envs": ("AZURE_OPENAI_API_KEY", "FOUNDRY_OPENAI_API_KEY",
                                    "ANTHROPIC_FOUNDRY_API_KEY"),
                       "price_in": 0.75, "price_out": 4.50},
}
DEFAULT_PROVIDER = "anthropic"
DEFAULT_DAILY_BUDGET = 20.0     # USD/day


def provider_key(provider: str) -> Optional[str]:
    for env in PROVIDERS[provider]["key_envs"]:
        v = os.getenv(env)
        if v:
            return v
    return None


def bedrock_creds() -> Optional[dict]:
    """Credentials for the Bedrock client, or None if not fully configured. A region is
    always required (AWS_REGION or BEDROCK_AWS_REGION). Then either method works:
      * a Bedrock API key (bearer token) in AWS_BEARER_TOKEN_BEDROCK — simplest; or
      * IAM SigV4 keys: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY (+ optional session token).
    The two are mutually exclusive at the SDK, so the bearer token wins when both are set."""
    region = (os.getenv("BEDROCK_AWS_REGION") or os.getenv("AWS_REGION")
              or os.getenv("AWS_DEFAULT_REGION"))
    if not region:
        return None
    token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    if token:
        return {"aws_region": region, "api_key": token}
    ak = os.getenv("AWS_ACCESS_KEY_ID")
    sk = os.getenv("AWS_SECRET_ACCESS_KEY")
    if ak and sk:
        return {"aws_region": region, "aws_access_key": ak, "aws_secret_key": sk,
                "aws_session_token": os.getenv("AWS_SESSION_TOKEN") or None}
    return None


def foundry_creds() -> Optional[dict]:
    """Credentials for the Foundry client, or None if not fully configured. Needs an
    endpoint — a resource name (ANTHROPIC_FOUNDRY_RESOURCE) or a full base URL
    (ANTHROPIC_FOUNDRY_BASE_URL, e.g. https://<resource>.services.ai.azure.com/anthropic/)
    — plus EITHER an API key (ANTHROPIC_FOUNDRY_API_KEY, simplest) OR Entra ID (no key,
    with the azure-identity package installed). The key wins when both are available."""
    resource = os.getenv("ANTHROPIC_FOUNDRY_RESOURCE")
    base_url = (os.getenv("ANTHROPIC_FOUNDRY_BASE_URL") or "").rstrip("/") or None
    if not (resource or base_url):
        return None
    key = provider_key("foundry")
    if key:
        return {"api_key": key, "resource": resource, "base_url": base_url}
    try:
        import azure.identity  # noqa: F401
    except Exception:
        return None
    return {"use_entra": True, "resource": resource, "base_url": base_url}


def foundry_openai_base_url() -> Optional[str]:
    """Foundry OpenAI v1 endpoint, or None. An explicit URL or resource wins; otherwise
    the Claude Foundry resource is reused (`…/anthropic` becomes `…/openai/v1`)."""
    explicit = (os.getenv("AZURE_OPENAI_BASE_URL") or os.getenv("FOUNDRY_OPENAI_BASE_URL") or "").strip().rstrip("/")
    if explicit:
        return explicit
    resource = (os.getenv("AZURE_OPENAI_RESOURCE") or os.getenv("FOUNDRY_OPENAI_RESOURCE") or "").strip()
    if resource:
        return f"https://{resource}.services.ai.azure.com/openai/v1"
    claude = (os.getenv("ANTHROPIC_FOUNDRY_BASE_URL") or "").strip().rstrip("/")
    if claude:
        if claude.endswith("/anthropic"):
            return claude[: -len("/anthropic")] + "/openai/v1"
        if "/openai/" in claude:
            return claude
        return claude + "/openai/v1"
    resource = (os.getenv("ANTHROPIC_FOUNDRY_RESOURCE") or "").strip()
    if resource:
        return f"https://{resource}.services.ai.azure.com/openai/v1"
    return None


def foundry_openai_creds() -> Optional[dict]:
    """Credentials for Foundry OpenAI, or None if not fully configured. Needs a v1
    endpoint (see foundry_openai_base_url) plus EITHER an API key (AZURE_OPENAI_API_KEY,
    or the Claude Foundry key on the same resource) OR Entra ID (azure-identity, no key).
    The key wins when both are available."""
    base_url = foundry_openai_base_url()
    if not base_url:
        return None
    key = provider_key("foundry_openai")
    if key:
        return {"api_key": key, "base_url": base_url}
    try:
        import azure.identity  # noqa: F401
    except Exception:
        return None
    return {"use_entra": True, "base_url": base_url}


def provider_configured(provider: str) -> bool:
    """Whether a provider has usable credentials — an API key, AWS creds for Bedrock,
    or a Foundry endpoint plus a key / Entra ID."""
    if provider == "bedrock":
        return bedrock_creds() is not None
    if provider == "foundry":
        return foundry_creds() is not None
    if provider == "foundry_openai":
        return foundry_openai_creds() is not None
    return provider_key(provider) is not None


# ---- budget / usage -----------------------------------------------------------

def budget(conn) -> float:
    try:
        return float(settings.get_setting(conn, "ai_daily_budget", str(DEFAULT_DAILY_BUDGET)))
    except (TypeError, ValueError):
        return DEFAULT_DAILY_BUDGET


def daily_spend(conn) -> float:
    day = db.now_iso()[:10]
    row = conn.execute(f"SELECT COALESCE(SUM(cost), 0) AS c FROM ai_usage WHERE day = {_P}",
                       (day,)).fetchone()
    return float(row["c"] or 0.0)


def log_usage(conn, cost, in_tok, out_tok, kind: str) -> None:
    ts = db.now_iso()
    conn.execute(
        f"INSERT INTO ai_usage (day, created_at, cost, input_tokens, output_tokens, kind) "
        f"VALUES ({_P},{_P},{_P},{_P},{_P},{_P})",
        (ts[:10], ts, cost or 0.0, in_tok or 0, out_tok or 0, kind))
    conn.commit()


def cfg_for(conn, provider: str, model: str, key: Optional[str] = None) -> dict:
    """An AI config for a run's fixed provider + model; prices from ai_models. `key` lets
    the caller supply the credential (the web app resolves it through its own patchable
    hook); by default it comes from the environment."""
    p = PROVIDERS.get(provider) or PROVIDERS[DEFAULT_PROVIDER]
    row = ai_models.find(conn, provider, model)
    price_in = row["input_per_m"] if row else p["price_in"]
    price_out = row["output_per_m"] if row else p["price_out"]
    cfg = {"provider": provider, "label": p["label"], "base_url": p["base_url"],
           "model": model or p["model"],
           "key": key if key is not None else provider_key(provider),
           "price_in": price_in, "price_out": price_out,
           "grid": dict(row) if row else None,
           "peak_bitmap": peak_schedule.get_bitmap(conn, provider)}
    if provider == "foundry":
        # The endpoint (resource name or full base URL) rides on the config so _client
        # needn't re-read env — and so callers can override it per config.
        fc = foundry_creds() or {}
        cfg["resource"] = fc.get("resource")
        if fc.get("base_url"):
            cfg["base_url"] = fc["base_url"]
        cfg["use_entra"] = bool(fc.get("use_entra"))
    if provider == "foundry_openai":
        fc = foundry_openai_creds() or {}
        if fc.get("base_url"):
            cfg["base_url"] = fc["base_url"]
        cfg["use_entra"] = bool(fc.get("use_entra"))
    return cfg


# ---- the request ----------------------------------------------------------------

def eval_max_tokens() -> int:
    """A generous max_tokens so a reasoning model has room to reason AND emit the short JSON
    verdict; override with AI_EVAL_MAX_TOKENS."""
    return int(os.getenv("AI_EVAL_MAX_TOKENS", "4096"))


def sdk_version() -> Optional[str]:
    try:
        import anthropic
        return getattr(anthropic, "__version__", None)
    except Exception:
        return None


# Model families that reject `temperature` / `top_p` / `top_k` (HTTP 400) — they think
# adaptively and only accept `thinking` / `effort`. Matched as substrings of the model id
# (Bedrock ids embed the same names).
_NO_SAMPLING = ("sonnet-5", "opus-5", "opus-4-7", "opus-4-8", "fable")


def supports_sampling(model_id: str) -> bool:
    """Whether `temperature` may be sent to this model."""
    m = (model_id or "").lower()
    return not any(tag in m for tag in _NO_SAMPLING)


def request_kwargs(config: dict, system: str, messages: list, max_tokens: int, *,
                   cache_system: bool = False, temperature: Optional[float] = None,
                   thinking: Optional[dict] = None, effort: Optional[str] = None) -> dict:
    """The messages.create(...) keyword arguments — pure, so the exact request a run makes is
    testable. Each sampling control is added only when given; `thinking`/`effort` only for
    Claude providers (anthropic / bedrock / foundry). Raises ValueError for a temperature on
    a model that rejects it."""
    kwargs: Dict = dict(model=config["model"], max_tokens=max_tokens, messages=list(messages))
    if (system or "").strip():
        # Cache the stable system prefix on Anthropic and Foundry (cache_control) so a run's
        # repeated instructions are cheap cache-reads after the first call. DeepSeek's endpoint
        # ignores cache_control but auto-caches the same prefix, so a plain string is enough there.
        if cache_system and config.get("provider") in ("anthropic", "foundry"):
            kwargs["system"] = [{"type": "text", "text": system.strip(),
                                 "cache_control": {"type": "ephemeral"}}]
        else:
            kwargs["system"] = system.strip()
    if temperature is not None:
        if not supports_sampling(config["model"]):
            raise ValueError(f"{config['model']} rejects `temperature` (it thinks adaptively); "
                             f"run it without sampling controls or pick a model that accepts them")
        kwargs["temperature"] = float(temperature)
    claude = config.get("provider") in ("anthropic", "bedrock", "foundry")
    if thinking is not None and claude:
        kwargs["thinking"] = dict(thinking)
    if effort is not None and claude:
        kwargs["output_config"] = {"effort": effort}
    return kwargs


def openai_request_kwargs(config: dict, system: str, messages: list, max_tokens: int, *,
                          temperature: Optional[float] = None,
                          effort: Optional[str] = None) -> dict:
    """chat.completions.create(...) kwargs for Foundry OpenAI. `max_tokens` is sent as
    `max_completion_tokens` (reasoning models reject `max_tokens`). `effort` maps to
    `reasoning_effort`; omitted means the model's default (none on gpt-5.4). Temperature
    is only legal while reasoning is off, so a non-none effort plus a temperature fails
    fast. `thinking` is Claude-only and is not sent."""
    built = []
    if (system or "").strip():
        built.append({"role": "system", "content": system.strip()})
    built.extend(messages)
    kwargs: Dict = {"model": config["model"], "max_completion_tokens": max_tokens,
                    "messages": built}
    reasoning = effort not in (None, "", "none")
    if temperature is not None:
        if not supports_sampling(config["model"]):
            raise ValueError(f"{config['model']} rejects `temperature` (it thinks adaptively); "
                             f"run it without sampling controls or pick a model that accepts them")
        if reasoning:
            raise ValueError(f"{config['model']} only accepts `temperature` when reasoning effort "
                             f"is none; drop the temperature or set effort to none")
        kwargs["temperature"] = float(temperature)
    if reasoning:
        kwargs["reasoning_effort"] = effort
    return kwargs


class _RefreshingToken:
    """Entra bearer that re-reads on each request. The OpenAI SDK interpolates api_key
    into the Authorization header per call, so __str__ is the refresh hook."""

    def __init__(self, provider):
        self._provider = provider

    def __str__(self) -> str:
        return self._provider()

    def __bool__(self) -> bool:
        return True


def _client(config: dict, timeout: float, max_retries: int):
    """The SDK client for a config, or an error dict."""
    if config.get("provider") == "foundry_openai":
        return _openai_client(config, timeout, max_retries)
    try:
        import anthropic
    except Exception:
        return None, {"fatal": True,
                      "error": "The 'anthropic' package is not installed. Run: pip install -r requirements.txt"}
    if config.get("provider") == "bedrock":
        creds = bedrock_creds()
        if not creds:
            return None, {"fatal": True, "error": "AWS Bedrock credentials not set. Set a region "
                          "(BEDROCK_AWS_REGION or AWS_REGION) plus EITHER a Bedrock API key "
                          "(AWS_BEARER_TOKEN_BEDROCK) OR AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY in "
                          "~/gov-uk-corpus.env, then restart."}
        BedrockClient = getattr(anthropic, "AnthropicBedrock", None)
        if BedrockClient is None:
            return None, {"fatal": True, "error": "This 'anthropic' build has no Bedrock support. "
                          "Run: pip install -U 'anthropic[bedrock]'"}
        bkw = dict(aws_region=creds["aws_region"], timeout=timeout, max_retries=max_retries)
        if creds.get("api_key"):     # Bedrock API key -> Authorization: Bearer
            bkw["api_key"] = creds["api_key"]
        else:                        # IAM SigV4 keys
            bkw["aws_access_key"] = creds["aws_access_key"]
            bkw["aws_secret_key"] = creds["aws_secret_key"]
            if creds.get("aws_session_token"):
                bkw["aws_session_token"] = creds["aws_session_token"]
        return BedrockClient(**bkw), None
    if config.get("provider") == "foundry":
        creds = foundry_creds() or {}
        # A config may carry its own endpoint (cfg_for stamps it); env is the fallback.
        resource = config.get("resource") or creds.get("resource")
        base_url = config.get("base_url") or creds.get("base_url") or None
        api_key = config.get("key") or creds.get("api_key")
        use_entra = config.get("use_entra") or creds.get("use_entra")
        if not (resource or base_url) or (not api_key and not use_entra):
            return None, {"fatal": True, "error": "Azure Foundry not configured. Set "
                          "ANTHROPIC_FOUNDRY_RESOURCE (or ANTHROPIC_FOUNDRY_BASE_URL) plus "
                          "ANTHROPIC_FOUNDRY_API_KEY in ~/gov-uk-corpus.env and restart — or "
                          "leave the key unset with azure-identity installed for Entra ID."}
        FoundryClient = getattr(anthropic, "AnthropicFoundry", None)
        if FoundryClient is None:
            return None, {"fatal": True, "error": "This 'anthropic' build has no Foundry support. "
                          "Run: pip install -U 'anthropic>=0.74'"}
        fkw = dict(timeout=timeout, max_retries=max_retries)
        if base_url:      # full endpoint URL and resource name are mutually exclusive
            fkw["base_url"] = base_url
        else:
            fkw["resource"] = resource
        if api_key:
            fkw["api_key"] = api_key
        else:              # Entra ID via the Azure token-provider pattern
            try:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            except Exception:
                return None, {"fatal": True, "error": "Entra ID needs the 'azure-identity' package. "
                              "Run: pip install azure-identity (or set ANTHROPIC_FOUNDRY_API_KEY)."}
            fkw["azure_ad_token_provider"] = get_bearer_token_provider(
                DefaultAzureCredential(), "https://ai.azure.com/.default")
        return FoundryClient(**fkw), None
    if not config.get("key"):
        return None, {"fatal": True, "error": f"No API key set for {config['label']}. Add its key to "
                      f"~/gov-uk-corpus.env and restart, or pick a provider that has one in Settings."}
    client_kwargs = dict(api_key=config["key"], timeout=timeout, max_retries=max_retries)
    if config.get("base_url"):               # empty => anthropic SDK default (Claude API)
        client_kwargs["base_url"] = config["base_url"]
    # Org-scoped ("Default") Anthropic keys need the workspace id header.
    ws = os.getenv("ANTHROPIC_WORKSPACE_ID")
    if ws and config["provider"] == "anthropic":
        client_kwargs["default_headers"] = {"anthropic-workspace-id": ws}
    return anthropic.Anthropic(**client_kwargs), None


def _openai_client(config: dict, timeout: float, max_retries: int):
    """OpenAI SDK client pointed at the Foundry v1 route, or an error dict."""
    try:
        from openai import OpenAI
    except Exception:
        return None, {"fatal": True,
                      "error": "The 'openai' package is not installed. Run: pip install -r requirements.txt"}
    creds = foundry_openai_creds() or {}
    base_url = config.get("base_url") or creds.get("base_url")
    api_key = config.get("key") or creds.get("api_key")
    use_entra = config.get("use_entra") or creds.get("use_entra")
    if not base_url or (not api_key and not use_entra):
        return None, {"fatal": True, "error": "Azure Foundry OpenAI not configured. Set "
                      "AZURE_OPENAI_BASE_URL (or AZURE_OPENAI_RESOURCE) plus AZURE_OPENAI_API_KEY "
                      "in ~/gov-uk-corpus.env and restart — or leave the key unset with azure-identity "
                      "installed for Entra ID. A Foundry resource already set for Claude "
                      "(ANTHROPIC_FOUNDRY_RESOURCE / ANTHROPIC_FOUNDRY_API_KEY) is reused when the "
                      "OpenAI vars are absent."}
    okw = dict(base_url=base_url, timeout=timeout, max_retries=max_retries)
    if api_key:
        okw["api_key"] = api_key
    else:
        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        except Exception:
            return None, {"fatal": True, "error": "Entra ID needs the 'azure-identity' package. "
                          "Run: pip install azure-identity (or set AZURE_OPENAI_API_KEY)."}
        okw["api_key"] = _RefreshingToken(get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"))
    return OpenAI(**okw), None


def _openai_text(msg) -> tuple:
    """(reply text, finish_reason) from a chat.completions payload."""
    choices = getattr(msg, "choices", None) or (msg.get("choices") if isinstance(msg, dict) else None) or []
    if not choices:
        return "", None
    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None and isinstance(choice, dict):
        message = choice.get("message")
    content = getattr(message, "content", None) if message is not None and not isinstance(message, dict) else (
        message.get("content") if isinstance(message, dict) else None)
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                parts.append(p.get("text") or "")
            else:
                parts.append(getattr(p, "text", "") or "")
        text = "".join(parts)
    else:
        text = ""
    if not text and message is not None:
        text = (getattr(message, "refusal", None) if not isinstance(message, dict)
                else message.get("refusal")) or ""
    stop = getattr(choice, "finish_reason", None) if not isinstance(choice, dict) else choice.get("finish_reason")
    return text, stop


def _priced(config: dict, text: str, actual_model, usage, stop_reason) -> dict:
    """The chat() success payload: tokens and cost from a normalised usage breakdown."""
    u = pricing.usage_breakdown(usage)   # {in_total, hit, miss, out} across providers
    peak = pricing.is_peak_now(config.get("peak_bitmap"))
    in_tok = u["in_total"] if u else None
    out_tok = u["out"] if u else None
    cache_hit = u["hit"] if u else 0
    cache_miss = u["miss"] if u else 0
    cost = None
    if u is not None:
        grid = config.get("grid")
        if grid:   # tiered price grid × peak schedule: miss tokens at miss rate, hit at hit rate
            cost = pricing.call_cost(grid, peak, u["miss"], u["out"], u["hit"])
        if cost is None:   # fall back to the flat standard rate on the total input
            cost = round((in_tok / 1e6) * config["price_in"]
                         + (out_tok / 1e6) * config["price_out"], 6)
    return {"reply": text, "model": config["model"], "actual_model": actual_model,
            "provider": config["provider"], "peak": peak, "stop_reason": stop_reason,
            "cache_hit_tokens": cache_hit, "cache_miss_tokens": cache_miss,
            "input_tokens": in_tok, "output_tokens": out_tok, "cost_usd": cost,
            "price_input_per_m": config["price_in"], "price_output_per_m": config["price_out"]}


def _call_failed(config: dict, e: Exception) -> dict:
    logging.getLogger("assistant").exception("AI call failed (provider=%s model=%s)",
                                             config.get("provider"), config.get("model"))
    return {"error": f"{type(e).__name__}: {e}"}


def chat(config: dict, system: str, messages: list, max_tokens: int = 1024,
         cache_system: bool = False, *, temperature: Optional[float] = None,
         thinking: Optional[dict] = None, effort: Optional[str] = None) -> dict:
    """One model call. Returns the reply text plus what it cost, or {"error": ...}
    ({"fatal": True} when no retry can help — missing key / bad config)."""
    # Retry transient errors (429 / 5xx / timeouts) at the SDK, honouring Retry-After, so a
    # single hiccup from a rate-limiting or slow provider self-heals instead of aborting an
    # evaluation run. Tune with AI_MAX_RETRIES / AI_TIMEOUT.
    timeout = float(os.getenv("AI_TIMEOUT", "45"))
    max_retries = int(os.getenv("AI_MAX_RETRIES", "4"))
    # Validate the request before building a client, so a refused temperature is fatal
    # even when the SDK or credentials are missing.
    if config.get("provider") == "foundry_openai":
        try:
            kwargs = openai_request_kwargs(config, system, messages, max_tokens,
                                           temperature=temperature, effort=effort)
        except ValueError as e:
            return {"fatal": True, "error": str(e)}
        client, err = _client(config, timeout, max_retries)
        if err:
            return err
        try:
            msg = client.chat.completions.create(**kwargs)
            text, stop = _openai_text(msg)
            return _priced(config, text, getattr(msg, "model", None), getattr(msg, "usage", None), stop)
        except Exception as e:
            return _call_failed(config, e)
    try:
        kwargs = request_kwargs(config, system, messages, max_tokens, cache_system=cache_system,
                                temperature=temperature, thinking=thinking, effort=effort)
    except ValueError as e:
        return {"fatal": True, "error": str(e)}
    client, err = _client(config, timeout, max_retries)
    if err:
        return err
    try:
        msg = client.messages.create(**kwargs)
        text = "".join(getattr(b, "text", "") for b in msg.content)
        return _priced(config, text, getattr(msg, "model", None), getattr(msg, "usage", None),
                       getattr(msg, "stop_reason", None))
    except Exception as e:  # network / auth / API errors surfaced to the caller
        return _call_failed(config, e)


def reply(config: dict, system: str, prompt: str, *, sampling: Optional[dict] = None) -> dict:
    """Single-turn convenience wrapper over chat (used by the page evaluator)."""
    s = sampling or {}
    return chat(config, system, [{"role": "user", "content": prompt}],
                max_tokens=eval_max_tokens(), temperature=s.get("temperature"),
                thinking=s.get("thinking"), effort=s.get("effort"))
