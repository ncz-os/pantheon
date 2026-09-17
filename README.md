# mnemos-pantheon

PANTHEON is the Mnemos **unified LLM facade** — one OpenAI-compatible surface in
front of every provider. It is a separately installable `mnemos.*` namespace
distribution (PEP 420) that overlays onto `mnemos-core`; it is not a monorepo
subpackage.

```bash
pip install mnemos-core mnemos-pantheon
```

PANTHEON is bundled in the Mnemos umbrella image (`ghcr.io/ncz-os/mnemos-enterprise`); when
the distribution is present, `mnemos-core` mounts its routes automatically.

The distribution keeps the original import paths through PEP 420 namespace
packaging:

- `mnemos.domain.pantheon`
- `mnemos.api.routes.pantheon`
- `mnemos.api.pantheon_shadow`

It does not own the shared namespace directories `mnemos/`, `mnemos/domain/`,
`mnemos/api/`, or `mnemos/api/routes/`. Those directories intentionally do not
contain `__init__.py` files here. The owned leaf package is
`mnemos/domain/pantheon/`, which keeps its `__init__.py`.

## Dependencies

This package depends on the core and sibling Mnemos distributions that provide
the shared namespace surface used by PANTHEON:

- `mnemos-core` for configuration, extras, numeric helpers, plan windows,
  provider registry, rate limiting, API dependencies, persistence, lifecycle,
  OpenAI compatibility helpers, and NATS publisher integration.
- `mnemos-graeae` for provider configuration, API keys, engines, and model
  registry integration.
- `mnemos-knemon` for budget evaluation.

PANTHEON also ships FastAPI routes and an OpenAI-compatible shadow app, so the
runtime dependencies include `fastapi`, `httpx`, `uvicorn`, and `gunicorn`.
NATS JetStream cooldown sharing is optional and is exposed as the `nats` extra.

## Deployment

Deployment artifacts live under `deploy/pantheon/`.

Development shadow launch:

```sh
uvicorn mnemos.api.pantheon_shadow:app --port 4101
```

Production worker-pool launch:

```sh
gunicorn mnemos.api.pantheon_shadow:app \
  -k uvicorn.workers.UvicornWorker \
  -w "${WEB_CONCURRENCY}" \
  -b 127.0.0.1:4110
```

See `deploy/pantheon/README.md` for the systemd unit, environment template, and
Caddy VIP cutover notes.

## Namespace Caveats

All Mnemos split distributions must use compatible PEP 420 namespace packaging.
Do not add `__init__.py` files at shared namespace levels in this package. A
different installed distribution that turns `mnemos` into a regular package can
hide namespace portions from sibling distributions unless it explicitly extends
the package path.


## Backend coverage

**The `model_registry` overlay behind `GET /pantheon/v1/models` is
PostgreSQL-only.** `_registry_rows()` reads `mnemos.core.lifecycle._pool`, which
core populates only for the PostgreSQL profile and sets to `None` everywhere
else, so on SQLite, MySQL, MariaDB, Oracle and Db2 it returns an empty list. The
registry-sourced fields — pricing, capabilities, context window, max output
tokens and arena rank — are then simply absent from `list_models()`.

The failure is quiet rather than loud: `/v1/models` still answers with a
catalog, because the GRAEAE engine and the JSON/SQLite cache paths survive. Two
things are lost. Models that also come from another source keep their entry but
lose every registry-sourced field, since the registry loop is what supplies
them. Models that exist **only** in `model_registry` are absent from the catalog
altogether — that loop is the only thing that would have added them. That
matters in practice, because the shipped
`deploy/docker-compose.mnemos-api.yml` runs the Oracle image, so this has been
the live configuration.

This is **not fixed**, and is not presented as fixed. It is now diagnosable: one
warning per process naming the active backend. Closing it properly means either
widening core's neutral `ConsultationAuditRepository.fetch_available_models`
(which returns three columns where this call site consumes seventeen, and which
risks regressing core's own `openai_compat` consumer) or adding a new ABC method
in core plus six backend implementations. Either is a core-owner decision.

## Build infrastructure & partners

Continuous integration and package distribution for this project are generously
supported by our open-source infrastructure partners:

- **[GitLab](https://gitlab.com/)** — canonical source hosting and CI pipelines
  (format / lint / test gates), via the
  [GitLab for Open Source](https://about.gitlab.com/solutions/open-source/) program.
- **[Buildkite](https://buildkite.com/)** — CI/CD orchestration with hosted macOS
  and Linux agents, and our APT package registry host
  (`packages.buildkite.com/ncz-os/ncz`), via the
  [Buildkite Open Source](https://buildkite.com/pricing) program.

Thank you to both for backing open-source software.


## Provider attempt accounting

Python 3.13 or newer is required, matching core. Development installs resolve
real sibling checkouts `../mnemos-core`, `../graeae`, and `../knemon`; runtime
packages are not replaced with test stubs.

The route-level budget check is advisory. Each actual HTTP retry/fallback,
embedding request, and opened stream atomically reserves KNEMON headroom for
its own provider and final bounded token allowance before upstream work.
Missing pricing or persistence fails closed under configured monetary caps.
Absent chat limits receive a 4096-token bound (reasoning models keep their
configured reasoning floor). UTF-8 request size conservatively estimates text
input; token usage determines settlement when present. Missing usage and
ambiguous failures retain the estimate. Subscription OAuth attempts are
recorded as `codex-oauth` with their actual reconciled model and zero marginal
API price; this is not a subscription-utilization quota guarantee.

Streaming usage and identity travel with the opened attempt and explicit
request state. Finalization runs on completion, cancellation, and error-body
read/close failures. The route records audit metadata; the attempt owns the
usage ledger, preventing duplicate rows. Requests delegated to GRAEAE are
accounted once in its `graeae` budget scope. Other PANTHEON attempts use the
`pantheon` scope. See KNEMON's README for recovery and rolling-window scaling
limits.
