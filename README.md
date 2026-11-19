# Trophy Rarity Forecast — GTA VI

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/fiqirealme52-max/TrophyForecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/fiqirealme52-max/TrophyForecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts how many Grand Theft Auto VI players on PlayStation 5 will have earned each trophy one week after launch, from Exophase's earn-rate reads. For every trophy it writes five quantiles (5%, 25%, 50%, 75%, 95%) of the earn rate at the case's target, scored by pinball loss on the logit scale.

This is an entry for [Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) on [make.fun](https://make.fun).

## How it forecasts

`main.ts` (Node 22, standard library only) reads a case (`/input/<id>.in`) and writes `/output/<id>.out`. Its median blends two views in logit space:

- **Log-linear**: least-squares fit of `logit(rate)` against `ln(1 + hours since release)` over the whole history (not just the last 24 hours), with slopes shrunk toward the median of the trophy's group (story, side, collectible, grind, activity, from the description). Reads showing an unchanged page are collapsed into one state so they don't fake precision.
- **Weibull adoption**: `rate(h) = F·(1 − exp(−((h−1)/τ)^k))`, a saturation curve for cumulative completion, with the ceiling fitted by closed-form least squares over a (τ, k) grid.

Hard facts the house baselines ignore are enforced as constraints: chapter order (Ch1 ≥ Ch2 ≥ Ch3 ≥ Ch4 ≥ final mission), and the platinum and 100%-completion gold trophies are no more common than any other trophy. The spread combines the fit residual, the slope's standard error times the extrapolation distance, the median's move, a refresh shock, and a model error term, clamped to [0.6, 1.6] logit units.

## Try it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`input/` holds the kit's practice cases (`p00.in`, `p01.in`). The program is deterministic, offline, and runs in well under the 8-second cap.

## License

MIT — see `LICENSE`.

New to make.fun? Sign up with my link: https://make.fun/r/vfa7ye18 (referral code `vfa7ye18`).
