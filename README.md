# mnemos-pantheon

PANTHEON is the Mnemos **unified LLM facade** — one OpenAI-compatible surface in
front of every provider. It is a separately installable `mnemos.*` namespace
distribution (PEP 420) that overlays onto `mnemos-core`; it is not a monorepo
subpackage.

```bash
pip install mnemos-core mnemos-pantheon
```

PANTHEON is bundled in the Mnemos umbrella image (`ghcr.io/ncz-os/mnemos`); when
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
