# 0005 — Foundry OpenAI is its own provider

- **Status:** accepted (2026-10-15)

## Context

Azure AI Foundry already serves Claude through the Anthropic SDK (`foundry`,
`AnthropicFoundry`, Messages API). The same resource also deploys OpenAI models.
`gpt-5.4` and `gpt-5.4-mini` speak Chat Completions on `/openai/v1/`, not Messages:
they want `max_completion_tokens`, report cache reads as
`prompt_tokens_details.cached_tokens`, and accept `temperature` only while
reasoning effort is `none` (the gpt-5.4 default).

## Decision

Add a `foundry_openai` provider that calls the OpenAI SDK. The deployment name is
the model id (`gpt-5.4`, `gpt-5.4-mini`). Credentials prefer `AZURE_OPENAI_*` and
otherwise reuse the Claude Foundry resource and key, rewriting an `/anthropic`
base URL to `/openai/v1`. Both models are seeded with OpenAI's published standard
rates (cached input at the cache-hit tier; no peak surcharge). An existing model
registry gains them on the next `seed_defaults` without overwriting rows already
stored under those ids.

## Consequences

- Claude Foundry stays on the Anthropic SDK. A Foundry outage or key still has to
  be configured once for both when they share a resource.
- `effort` on a Foundry OpenAI call is `reasoning_effort`. Combining it with
  `temperature` fails before the request. Product runs send neither, so gpt-5.4
  stays on its default of no reasoning.
- gpt-5.4's >272k-input tariff is not applied. Page-sized eval prompts stay under
  that; a long-context run would under-bill until the grid is edited.
