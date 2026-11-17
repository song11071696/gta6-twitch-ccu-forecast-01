# GTA VI Twitch CCU Forecaster — make.fun gta6-twitch-ccu-forecast-01

Forecasts, hour by hour, how many people watch Grand Theft Auto VI live on
Twitch over the 24 UTC hours of its release day (2026-11-19), from hour 1's
print onward, as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) per hour.

## Method (one paragraph)

A launch-day Twitch curve is a shape times a scale. Hour 1's print fixes the
scale; the shape — each later hour's viewers as a ratio to hour 1's — is
learned from the case's comparables (past big launches' release-day curves,
read at run time from the case file). The shape depends on scale: a launch
already at 100k+ viewers in hour 1 stays near flat (median hour-19 ratio
~1.25), while a small launch can grow 100x by evening. So each comparable is
weighted by how close its hour-1 scale is to this case's (Gaussian kernel on
log scale, bandwidth 1.0), and each hour's five quantiles are the weighted
quantiles of the comparables' log-ratios. Full write-up: [METHOD.md](METHOD.md).

## Validation (leave-one-out, on the 37 usable comparables in the kit)

| model | mean pinball loss | wins vs baseline |
|---|---|---|
| This model (scale-weighted quantiles) | **0.405** | **32 / 37** |
| House baseline (persistence, σ=0.15) | 0.930 | — |

56.5% lower loss than the house baseline. Bandwidth and spread were tuned
by the same leave-one-out grid (bandwidth 1.0, no spread shrinkage optimal).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Standard library only, Python 3.13. Reads every `<id>.in` in the input
directory, writes `<id>.out` (`{"hours": [{"hour": 2, "q": [q05, q25, q50,
q75, q95]}, ...]}`, 23 rows, hours 2..24) to the output directory.

## Files

- `main.py` — the forecaster (the entry).
- `METHOD.md` — full method, data, validation, and limitations.
- `LICENSE` — MIT.

## make.fun

- Competition: [gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
- Entrant: [@lucky-quasar-40](https://make.fun/r/p2453ks7)
