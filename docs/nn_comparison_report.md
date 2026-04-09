# BEQ Filter Prediction — Comparison Report

Generated from 241 unique titles with real extracted LFE audio,
compared against hand-coded BEQ catalogue entries.

## Summary

| Metric | Value |
|---|---|
| Model | Late fusion XGBoost (α=0.3), one-hot filter types |
| Training data | 7993 synthetic catalogue entries (no real audio needed) |
| Validation titles | 241 unique titles with real LFE WAVs |
| Mean downstream error | **3.01 dB** (20-80 Hz band) |
| Under 2 dB (expert quality) | 82/241 (34%) |
| Under 3 dB (good starting point) | 146/241 (60%) |
| Under 5 dB (usable) | 206/241 (85%) |
| Report generation time | 181s |

## Per-author performance

| Author | Titles | Mean | <2 dB | >4 dB |
|---|---|---|---|---|
| mobe1969 | 106 | 3.58 dB | 22 | 35 |
| aron7awol | 64 | 2.13 dB | 33 | 5 |
| kaelaria | 38 | 3.15 dB | 13 | 10 |
| remixmark | 14 | 3.20 dB | 5 | 4 |
| t1g8rsfan | 11 | 2.55 dB | 4 | 0 |
| halcyon888 | 5 | 1.74 dB | 4 | 0 |
| mikejl | 3 | 2.88 dB | 1 | 1 |

## Best predictions — under 2 dB (82 titles)

These match expert quality. The predicted filter chain produces a
frequency response within 2 dB of the hand-coded catalogue entry
across the 20-80 Hz bass extension band.

### Mindhunter (2017) — 0.40 dB — by aron7awol
- **Predicted (6 filters)**: LowShelf(41Hz, +1.2dB, Q=1.0), LowShelf(22Hz, +2.8dB, Q=1.0), LowShelf(19Hz, +3.4dB, Q=0.8), LowShelf(18Hz, +2.6dB, Q=0.8), LowShelf(7Hz, +1.7dB, Q=0.4), LowShelf(5Hz, +0.9dB, Q=0.2)
- **Catalogue (4 filters)**: PeakingEQ(50Hz, +0.5dB, Q=1.0), LowShelf(22Hz, +5.0dB, Q=0.9), LowShelf(22Hz, +5.0dB, Q=0.9), LowShelf(10Hz, +4.0dB, Q=0.9)

### South Park (1997) — 0.44 dB — by remixmark
- **Predicted (6 filters)**: LowShelf(19Hz, -0.9dB, Q=1.7), LowShelf(27Hz, +3.0dB, Q=0.9), LowShelf(32Hz, +2.7dB, Q=1.3), LowShelf(37Hz, +3.3dB, Q=0.9), LowShelf(33Hz, +2.1dB, Q=0.7), LowShelf(13Hz, +0.7dB, Q=0.5)
- **Catalogue (4 filters)**: LowShelf(14Hz, +5.3dB, Q=0.9), LowShelf(30Hz, +7.0dB, Q=0.9), LowShelf(30Hz, +7.0dB, Q=0.9), PeakingEQ(70Hz, +0.6dB, Q=0.8)

### Arcane (2021) — 0.49 dB — by aron7awol
- **Predicted (5 filters)**: LowShelf(37Hz, +2.8dB, Q=1.1), LowShelf(27Hz, +2.5dB, Q=1.1), LowShelf(27Hz, +2.7dB, Q=1.0), LowShelf(18Hz, +1.7dB, Q=0.5), LowShelf(8Hz, +1.3dB, Q=0.2)
- **Catalogue (5 filters)**: PeakingEQ(12Hz, -0.8dB, Q=6.0), LowShelf(15Hz, -4.3dB, Q=1.1), LowShelf(29Hz, +4.3dB, Q=0.9), LowShelf(29Hz, +4.3dB, Q=0.9), PeakingEQ(70Hz, +0.2dB, Q=0.9)

### The Raid 2 (2014) — 0.51 dB — by aron7awol
- **Predicted (6 filters)**: PeakingEQ(50Hz, +0.7dB, Q=1.0), LowShelf(25Hz, +2.9dB, Q=1.0), LowShelf(26Hz, +3.5dB, Q=1.1), LowShelf(21Hz, +2.9dB, Q=0.9), LowShelf(11Hz, +2.2dB, Q=0.7), LowShelf(5Hz, +0.8dB, Q=0.5)
- **Catalogue (3 filters)**: PeakingEQ(50Hz, +0.5dB, Q=1.0), LowShelf(20Hz, +6.4dB, Q=0.9), LowShelf(20Hz, +6.4dB, Q=0.9)

### Tropic Thunder (2008) — 0.68 dB — by aron7awol
- **Predicted (6 filters)**: PeakingEQ(46Hz, +1.0dB, Q=1.1), LowShelf(26Hz, +2.7dB, Q=1.1), LowShelf(21Hz, +3.9dB, Q=0.9), LowShelf(19Hz, +3.4dB, Q=0.8), LowShelf(13Hz, +2.6dB, Q=0.6), LowShelf(7Hz, +1.4dB, Q=0.2)
- **Catalogue (6 filters)**: PeakingEQ(51Hz, +0.2dB, Q=1.0), PeakingEQ(32Hz, -1.1dB, Q=5.0), LowShelf(20Hz, +3.6dB, Q=0.8), LowShelf(20Hz, +3.6dB, Q=0.8), LowShelf(20Hz, +3.6dB, Q=0.8), LowShelf(20Hz, +3.6dB, Q=0.8)

### Transformers: The Last Knight (2017) — 0.68 dB — by aron7awol
- **Predicted (6 filters)**: PeakingEQ(42Hz, +0.7dB, Q=1.1), LowShelf(23Hz, +2.3dB, Q=1.3), LowShelf(22Hz, +3.5dB, Q=0.9), LowShelf(16Hz, +3.2dB, Q=0.9), LowShelf(8Hz, +2.2dB, Q=0.4), LowShelf(5Hz, +0.9dB, Q=0.3)
- **Catalogue (6 filters)**: PeakingEQ(35Hz, +1.4dB, Q=1.0), LowShelf(10Hz, +5.0dB, Q=0.9), LowShelf(10Hz, +5.0dB, Q=0.9), LowShelf(19Hz, +3.6dB, Q=1.1), LowShelf(19Hz, +3.6dB, Q=1.1), LowShelf(19Hz, +3.6dB, Q=1.1)

### Foundation (2021) — 0.72 dB — by aron7awol
- **Predicted (6 filters)**: LowShelf(28Hz, +2.3dB, Q=1.1), LowShelf(23Hz, +3.4dB, Q=0.8), LowShelf(27Hz, +2.9dB, Q=0.8), LowShelf(20Hz, +1.9dB, Q=0.7), LowShelf(11Hz, +1.3dB, Q=0.4), LowShelf(5Hz, +0.7dB, Q=0.2)
- **Catalogue (5 filters)**: LowShelf(19Hz, +6.7dB, Q=0.9), LowShelf(19Hz, +6.7dB, Q=0.9), LowShelf(19Hz, +6.7dB, Q=0.9), LowShelf(19Hz, +6.7dB, Q=0.9), PeakingEQ(42Hz, +1.1dB, Q=0.8)

### Castlevania (2017) — 0.75 dB — by aron7awol
- **Predicted (5 filters)**: LowShelf(32Hz, +2.9dB, Q=1.3), LowShelf(18Hz, +4.1dB, Q=1.2), LowShelf(20Hz, +2.8dB, Q=1.4), LowShelf(18Hz, +2.4dB, Q=0.6), LowShelf(10Hz, +1.4dB, Q=0.5)
- **Catalogue (4 filters)**: PeakingEQ(55Hz, +0.4dB, Q=1.0), LowShelf(23Hz, +5.0dB, Q=0.9), LowShelf(23Hz, +5.0dB, Q=0.9), LowShelf(10Hz, -5.0dB, Q=0.8)

### Planet Earth II (2016) — 0.75 dB — by aron7awol
- **Predicted (6 filters)**: LowShelf(39Hz, +1.0dB, Q=1.1), LowShelf(23Hz, +2.5dB, Q=1.0), LowShelf(20Hz, +3.1dB, Q=0.8), LowShelf(16Hz, +2.3dB, Q=0.7), LowShelf(7Hz, +1.7dB, Q=0.4), LowShelf(5Hz, +1.0dB, Q=0.2)
- **Catalogue (5 filters)**: LowShelf(22Hz, +5.0dB, Q=0.9), LowShelf(22Hz, +5.0dB, Q=0.9), LowShelf(10Hz, +4.0dB, Q=0.9), LowShelf(10Hz, +4.0dB, Q=0.9), LowShelf(10Hz, +4.0dB, Q=0.9)

### Inside Out (2015) — 0.76 dB — by aron7awol
- **Predicted (6 filters)**: PeakingEQ(39Hz, +2.4dB, Q=1.3), LowShelf(24Hz, +2.7dB, Q=1.9), LowShelf(20Hz, +4.1dB, Q=1.2), LowShelf(19Hz, +2.5dB, Q=1.5), LowShelf(9Hz, +1.7dB, Q=0.6), LowShelf(6Hz, +0.6dB, Q=0.1)
- **Catalogue (6 filters)**: PeakingEQ(55Hz, +0.2dB, Q=1.0), PeakingEQ(20Hz, -2.0dB, Q=8.0), LowShelf(18Hz, +5.0dB, Q=0.8), LowShelf(18Hz, +5.0dB, Q=0.8), LowShelf(18Hz, +5.0dB, Q=0.8), LowShelf(10Hz, +5.0dB, Q=0.9)

### Primal (2019) — 0.79 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(26Hz, +3.4dB, Q=1.1), LowShelf(19Hz, +2.9dB, Q=1.1), LowShelf(16Hz, +3.1dB, Q=1.0), LowShelf(17Hz, +2.6dB, Q=0.9), LowShelf(10Hz, +2.1dB, Q=0.9), LowShelf(6Hz, +1.4dB, Q=0.8)
- **Catalogue (4 filters)**: LowShelf(10Hz, +1.7dB, Q=1.3), LowShelf(10Hz, +1.7dB, Q=1.3), PeakingEQ(13Hz, +4.0dB, Q=3.0), PeakingEQ(22Hz, +3.0dB, Q=3.0)

### The Platform (2019) — 0.80 dB — by aron7awol
- **Predicted (6 filters)**: PeakingEQ(49Hz, +0.7dB, Q=1.0), LowShelf(25Hz, +2.9dB, Q=1.3), LowShelf(21Hz, +4.1dB, Q=0.9), LowShelf(19Hz, +3.3dB, Q=0.8), LowShelf(10Hz, +2.6dB, Q=0.6), LowShelf(7Hz, +1.7dB, Q=0.4)
- **Catalogue (2 filters)**: LowShelf(26Hz, +4.3dB, Q=0.7), LowShelf(26Hz, +4.3dB, Q=0.7)

### Drive (2011) — 0.84 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(32Hz, +3.2dB, Q=1.3), LowShelf(25Hz, +2.1dB, Q=1.1), LowShelf(18Hz, +3.6dB, Q=1.0), LowShelf(16Hz, +3.6dB, Q=0.9), LowShelf(13Hz, +3.1dB, Q=1.0), LowShelf(9Hz, +2.2dB, Q=0.7)
- **Catalogue (5 filters)**: PeakingEQ(10Hz, +2.0dB, Q=2.0), PeakingEQ(20Hz, +3.0dB, Q=3.0), LowShelf(22Hz, +2.5dB, Q=0.8), LowShelf(22Hz, +2.5dB, Q=0.8), LowShelf(22Hz, +2.5dB, Q=0.8)

### Pantheon (2022) — 0.89 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(25Hz, +1.8dB, Q=1.1), LowShelf(21Hz, +2.8dB, Q=1.0), LowShelf(19Hz, +3.5dB, Q=1.0), LowShelf(21Hz, +3.3dB, Q=1.1), LowShelf(14Hz, +2.9dB, Q=1.0), LowShelf(9Hz, +1.9dB, Q=0.8)
- **Catalogue (4 filters)**: LowShelf(19Hz, +3.8dB, Q=0.8), LowShelf(19Hz, +3.8dB, Q=0.8), LowShelf(19Hz, +3.8dB, Q=0.8), LowShelf(19Hz, +3.8dB, Q=0.8)

### Cosmopolis (2012) — 0.90 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(21Hz, +4.4dB, Q=1.1), LowShelf(18Hz, +4.5dB, Q=1.0), LowShelf(17Hz, +3.0dB, Q=1.0), LowShelf(22Hz, +2.1dB, Q=1.0), LowShelf(9Hz, +1.7dB, Q=0.8), PeakingEQ(6Hz, +0.8dB, Q=0.6)
- **Catalogue (4 filters)**: LowShelf(15Hz, +2.7dB, Q=0.8), LowShelf(15Hz, +2.7dB, Q=0.8), LowShelf(15Hz, +2.7dB, Q=0.8), LowShelf(15Hz, +2.7dB, Q=0.8)

### Klaus (2019) — 0.93 dB — by aron7awol
- **Predicted (6 filters)**: PeakingEQ(48Hz, +0.9dB, Q=0.9), LowShelf(25Hz, +3.1dB, Q=1.1), LowShelf(20Hz, +4.0dB, Q=0.9), LowShelf(20Hz, +2.9dB, Q=0.8), LowShelf(7Hz, +2.2dB, Q=0.4), LowShelf(5Hz, +1.5dB, Q=0.3)
- **Catalogue (3 filters)**: PeakingEQ(45Hz, +0.7dB, Q=1.0), LowShelf(23Hz, +6.0dB, Q=1.1), LowShelf(10Hz, +5.0dB, Q=0.9)

### The Spy (2019) — 0.93 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(26Hz, +2.4dB, Q=1.1), LowShelf(21Hz, +2.9dB, Q=1.0), LowShelf(19Hz, +3.6dB, Q=1.0), LowShelf(21Hz, +3.4dB, Q=1.0), LowShelf(14Hz, +2.9dB, Q=1.0), LowShelf(7Hz, +1.6dB, Q=0.7)
- **Catalogue (5 filters)**: PeakingEQ(16Hz, +3.0dB, Q=2.0), LowShelf(21Hz, +2.5dB, Q=0.8), LowShelf(21Hz, +2.5dB, Q=0.8), LowShelf(21Hz, +2.5dB, Q=0.8), LowShelf(21Hz, +2.5dB, Q=0.8)

### The Wolf of Wall Street (2013) — 0.93 dB — by aron7awol
- **Predicted (6 filters)**: LowShelf(46Hz, +2.2dB, Q=0.9), LowShelf(22Hz, +3.3dB, Q=0.8), LowShelf(21Hz, +4.4dB, Q=0.8), LowShelf(22Hz, +3.2dB, Q=0.9), LowShelf(16Hz, +2.0dB, Q=1.3), LowShelf(17Hz, +1.4dB, Q=0.7)
- **Catalogue (3 filters)**: PeakingEQ(65Hz, +0.7dB, Q=1.0), LowShelf(31Hz, +4.6dB, Q=1.0), LowShelf(31Hz, +4.6dB, Q=1.0)

### Band of Brothers (2001) — 0.97 dB — by aron7awol
- **Predicted (6 filters)**: PeakingEQ(57Hz, +1.6dB, Q=1.3), LowShelf(27Hz, +4.7dB, Q=1.0), LowShelf(25Hz, +4.9dB, Q=0.6), LowShelf(26Hz, +4.2dB, Q=0.9), LowShelf(17Hz, +3.7dB, Q=0.6), LowShelf(14Hz, +2.2dB, Q=0.8)
- **Catalogue (5 filters)**: PeakingEQ(80Hz, +0.2dB, Q=1.0), LowShelf(13Hz, +4.4dB, Q=0.9), LowShelf(13Hz, +4.4dB, Q=0.9), LowShelf(29Hz, +6.0dB, Q=0.8), LowShelf(29Hz, +6.0dB, Q=0.8)

### Planet Earth III (2023) — 0.99 dB — by halcyon888
- **Predicted (6 filters)**: LowShelf(26Hz, +0.8dB, Q=1.1), LowShelf(20Hz, +2.3dB, Q=1.0), LowShelf(20Hz, +3.1dB, Q=1.3), LowShelf(21Hz, +2.5dB, Q=1.1), LowShelf(11Hz, +1.7dB, Q=1.0), LowShelf(12Hz, +1.3dB, Q=0.6)
- **Catalogue (2 filters)**: LowShelf(17Hz, +4.0dB, Q=0.8), PeakingEQ(22Hz, -0.6dB, Q=2.0)

### Solo Leveling (2024) — 0.99 dB — by remixmark
- **Predicted (6 filters)**: LowShelf(22Hz, +1.8dB, Q=1.3), LowShelf(24Hz, +4.4dB, Q=1.1), LowShelf(27Hz, +4.4dB, Q=1.0), LowShelf(28Hz, +3.2dB, Q=1.0), LowShelf(16Hz, +2.2dB, Q=0.7), LowShelf(17Hz, +1.3dB, Q=0.5)
- **Catalogue (4 filters)**: LowShelf(10Hz, +8.3dB, Q=0.9), LowShelf(23Hz, +8.2dB, Q=0.9), LowShelf(23Hz, +8.2dB, Q=0.9), PeakingEQ(60Hz, +0.6dB, Q=0.8)

### Prospect (2018) — 1.00 dB — by aron7awol
- **Predicted (5 filters)**: LowShelf(34Hz, +2.3dB, Q=1.2), LowShelf(21Hz, +3.8dB, Q=1.4), LowShelf(27Hz, +4.0dB, Q=1.6), LowShelf(26Hz, +2.8dB, Q=1.3), LowShelf(12Hz, +1.5dB, Q=0.4)
- **Catalogue (7 filters)**: PeakingEQ(80Hz, +0.7dB, Q=2.0), LowShelf(16Hz, +4.0dB, Q=0.9), LowShelf(16Hz, +4.0dB, Q=0.9), LowShelf(16Hz, +4.0dB, Q=0.9), LowShelf(30Hz, +7.0dB, Q=0.9), LowShelf(30Hz, +7.0dB, Q=0.9), LowShelf(52Hz, -5.0dB, Q=0.9)

### X-Men '97 (2024) — 1.02 dB — by t1g8rsfan
- **Predicted (5 filters)**: LowShelf(16Hz, +3.6dB, Q=1.1), LowShelf(20Hz, +4.3dB, Q=1.1), LowShelf(28Hz, +3.6dB, Q=1.8), PeakingEQ(37Hz, +2.2dB, Q=1.7), LowShelf(15Hz, +1.6dB, Q=0.4)
- **Catalogue (4 filters)**: LowShelf(10Hz, +3.0dB, Q=0.7), LowShelf(21Hz, +4.8dB, Q=1.0), LowShelf(21Hz, +4.8dB, Q=1.0), PeakingEQ(39Hz, +0.7dB, Q=0.7)

### Roma (2018) — 1.04 dB — by aron7awol
- **Predicted (5 filters)**: LowShelf(23Hz, +2.4dB, Q=1.4), LowShelf(20Hz, +4.0dB, Q=1.0), LowShelf(15Hz, +3.0dB, Q=0.7), LowShelf(10Hz, +2.3dB, Q=0.6), LowShelf(5Hz, +1.6dB, Q=0.3)
- **Catalogue (5 filters)**: PeakingEQ(30Hz, +0.8dB, Q=1.0), LowShelf(12Hz, +5.0dB, Q=0.9), LowShelf(12Hz, +5.0dB, Q=0.9), LowShelf(12Hz, +5.0dB, Q=0.9), LowShelf(12Hz, +5.0dB, Q=0.9)

### The Thursday Murder Club (2025) — 1.04 dB — by mikejl
- **Predicted (6 filters)**: LowShelf(29Hz, +3.0dB, Q=1.4), LowShelf(25Hz, +4.4dB, Q=1.4), LowShelf(23Hz, +4.5dB, Q=1.0), LowShelf(31Hz, +2.7dB, Q=1.5), PeakingEQ(33Hz, +2.2dB, Q=1.2), LowShelf(22Hz, +1.3dB, Q=0.8)
- **Catalogue (10 filters)**: PeakingEQ(13Hz, +2.2dB, Q=3.0), LowShelf(15Hz, +8.0dB, Q=0.8), LowShelf(15Hz, +8.0dB, Q=0.8), LowShelf(25Hz, +7.0dB, Q=0.8), LowShelf(25Hz, +7.0dB, Q=0.8), LowShelf(25Hz, +7.0dB, Q=0.8), PeakingEQ(28Hz, +7.0dB, Q=4.0), PeakingEQ(31Hz, -8.6dB, Q=2.4), PeakingEQ(60Hz, +0.3dB, Q=3.0), PeakingEQ(90Hz, +0.2dB, Q=1.0)

### Hercules (1997) — 1.08 dB — by remixmark
- **Predicted (5 filters)**: LowShelf(21Hz, +1.0dB, Q=1.4), LowShelf(28Hz, +5.5dB, Q=1.2), LowShelf(31Hz, +6.8dB, Q=0.9), LowShelf(33Hz, +4.9dB, Q=1.1), PeakingEQ(32Hz, +2.4dB, Q=1.5)
- **Catalogue (5 filters)**: LowShelf(27Hz, +6.4dB, Q=0.9), LowShelf(27Hz, +6.4dB, Q=0.9), LowShelf(27Hz, +6.4dB, Q=0.9), LowShelf(27Hz, +6.4dB, Q=0.9), PeakingEQ(60Hz, +1.0dB, Q=0.7)

### The Garfield Movie (2024) — 1.10 dB — by remixmark
- **Predicted (6 filters)**: LowShelf(29Hz, +1.5dB, Q=1.0), LowShelf(22Hz, +3.1dB, Q=1.2), LowShelf(25Hz, +4.5dB, Q=0.9), LowShelf(27Hz, +3.8dB, Q=1.2), LowShelf(18Hz, +2.3dB, Q=0.8), LowShelf(14Hz, +1.2dB, Q=0.5)
- **Catalogue (5 filters)**: LowShelf(10Hz, -3.1dB, Q=1.1), LowShelf(21Hz, +6.1dB, Q=0.9), LowShelf(21Hz, +6.1dB, Q=0.9), LowShelf(21Hz, +6.1dB, Q=0.9), PeakingEQ(55Hz, +0.6dB, Q=0.8)

### Mad Max: Fury Road (2015) — 1.11 dB — by aron7awol
- **Predicted (6 filters)**: PeakingEQ(44Hz, +0.6dB, Q=1.1), LowShelf(23Hz, +2.5dB, Q=1.3), LowShelf(19Hz, +4.0dB, Q=1.1), LowShelf(18Hz, +3.5dB, Q=0.9), LowShelf(11Hz, +2.9dB, Q=0.6), LowShelf(8Hz, +1.7dB, Q=0.5)
- **Catalogue (5 filters)**: PeakingEQ(11Hz, -5.0dB, Q=5.0), PeakingEQ(50Hz, +0.9dB, Q=1.0), LowShelf(27Hz, +3.5dB, Q=1.1), LowShelf(27Hz, +3.5dB, Q=1.1), LowShelf(10Hz, +5.0dB, Q=0.9)

### The Sympathizer (2024) — 1.12 dB — by kaelaria
- **Predicted (6 filters)**: LowShelf(27Hz, +4.0dB, Q=1.0), LowShelf(24Hz, +4.6dB, Q=1.1), LowShelf(23Hz, +4.2dB, Q=1.1), LowShelf(21Hz, +3.0dB, Q=1.2), LowShelf(15Hz, +1.3dB, Q=1.0), LowShelf(8Hz, +0.6dB, Q=0.6)
- **Catalogue (6 filters)**: PeakingEQ(13Hz, -6.0dB, Q=4.0), PeakingEQ(16Hz, -3.0dB, Q=4.0), LowShelf(20Hz, +8.0dB, Q=0.8), LowShelf(20Hz, +8.0dB, Q=0.8), LowShelf(20Hz, +8.0dB, Q=0.8), PeakingEQ(31Hz, -2.7dB, Q=3.0)

### Zero Dark Thirty (2012) — 1.13 dB — by aron7awol
- **Predicted (5 filters)**: PeakingEQ(42Hz, +2.2dB, Q=1.4), LowShelf(24Hz, +3.8dB, Q=1.5), LowShelf(28Hz, +3.7dB, Q=1.4), LowShelf(21Hz, +3.1dB, Q=0.8), LowShelf(15Hz, +1.9dB, Q=0.6)
- **Catalogue (5 filters)**: PeakingEQ(60Hz, +0.8dB, Q=1.0), LowShelf(32Hz, +3.8dB, Q=1.1), LowShelf(32Hz, +3.8dB, Q=1.1), LowShelf(12Hz, -3.2dB, Q=0.9), LowShelf(12Hz, -3.2dB, Q=0.9)

### The Lego Batman Movie (2017) — 1.15 dB — by aron7awol
- **Predicted (6 filters)**: PeakingEQ(38Hz, +1.1dB, Q=1.2), LowShelf(24Hz, +2.8dB, Q=1.1), LowShelf(17Hz, +4.3dB, Q=0.9), LowShelf(18Hz, +3.1dB, Q=0.9), LowShelf(7Hz, +2.1dB, Q=0.4), LowShelf(5Hz, +1.1dB, Q=0.3)
- **Catalogue (7 filters)**: PeakingEQ(42Hz, +1.9dB, Q=1.0), LowShelf(21Hz, +6.2dB, Q=1.0), LowShelf(21Hz, +6.2dB, Q=1.0), LowShelf(21Hz, +6.2dB, Q=1.0), LowShelf(15Hz, +5.0dB, Q=0.9), LowShelf(15Hz, +5.0dB, Q=0.9), LowShelf(15Hz, +5.0dB, Q=0.9)

### Princess Mononoke (1997) — 1.16 dB — by mobe1969
- **Predicted (6 filters)**: PeakingEQ(33Hz, +3.7dB, Q=1.7), LowShelf(27Hz, +3.7dB, Q=1.2), LowShelf(20Hz, +5.0dB, Q=1.2), LowShelf(19Hz, +4.8dB, Q=1.1), LowShelf(17Hz, +4.1dB, Q=1.3), LowShelf(14Hz, +3.6dB, Q=1.0)
- **Catalogue (5 filters)**: LowShelf(10Hz, +3.0dB, Q=0.8), LowShelf(10Hz, +3.0dB, Q=0.8), LowShelf(24Hz, +4.8dB, Q=0.8), LowShelf(24Hz, +4.8dB, Q=0.8), LowShelf(24Hz, +4.8dB, Q=0.8)

### Despicable Me 2 (2013) — 1.16 dB — by aron7awol
- **Predicted (6 filters)**: LowShelf(34Hz, +2.3dB, Q=2.0), LowShelf(19Hz, +1.9dB, Q=2.1), LowShelf(27Hz, +2.6dB, Q=1.1), LowShelf(21Hz, +2.6dB, Q=1.1), LowShelf(12Hz, +1.5dB, Q=0.5), LowShelf(6Hz, -1.1dB, Q=0.7)
- **Catalogue (4 filters)**: PeakingEQ(45Hz, +1.3dB, Q=1.1), LowShelf(10Hz, +4.4dB, Q=0.9), LowShelf(10Hz, +4.4dB, Q=0.9), LowShelf(26Hz, +6.0dB, Q=1.3)

### The Penguin (2024) — 1.18 dB — by kaelaria
- **Predicted (6 filters)**: LowShelf(31Hz, +4.0dB, Q=1.2), LowShelf(26Hz, +2.5dB, Q=1.6), LowShelf(19Hz, +4.2dB, Q=1.5), LowShelf(18Hz, +3.9dB, Q=1.0), LowShelf(15Hz, +2.9dB, Q=0.9), PeakingEQ(12Hz, +1.2dB, Q=0.9)
- **Catalogue (5 filters)**: LowShelf(19Hz, +7.6dB, Q=0.8), LowShelf(19Hz, +7.6dB, Q=0.8), LowShelf(19Hz, +7.6dB, Q=0.8), LowShelf(19Hz, +7.6dB, Q=0.8), PeakingEQ(30Hz, -3.4dB, Q=3.0)

### Sausage Party (2016) — 1.19 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(26Hz, +2.9dB, Q=1.0), LowShelf(20Hz, +3.1dB, Q=1.0), LowShelf(18Hz, +3.7dB, Q=1.1), LowShelf(20Hz, +3.3dB, Q=1.3), LowShelf(14Hz, +2.9dB, Q=1.1), LowShelf(10Hz, +1.8dB, Q=1.0)
- **Catalogue (5 filters)**: LowShelf(10Hz, +4.0dB, Q=0.8), LowShelf(18Hz, +5.0dB, Q=0.8), LowShelf(18Hz, +5.0dB, Q=0.8), LowShelf(18Hz, +5.0dB, Q=0.8), LowShelf(18Hz, +5.0dB, Q=0.8)

### Godzilla vs. Kong (2021) — 1.25 dB — by aron7awol
- **Predicted (6 filters)**: LowShelf(28Hz, +1.5dB, Q=1.2), LowShelf(23Hz, +3.2dB, Q=1.0), LowShelf(21Hz, +4.2dB, Q=1.0), LowShelf(26Hz, +3.3dB, Q=1.0), LowShelf(15Hz, +2.1dB, Q=0.6), PeakingEQ(11Hz, +1.0dB, Q=0.4)
- **Catalogue (6 filters)**: LowShelf(14Hz, +5.5dB, Q=1.1), LowShelf(14Hz, +5.5dB, Q=1.1), LowShelf(19Hz, +6.3dB, Q=0.8), LowShelf(19Hz, +6.3dB, Q=0.8), LowShelf(19Hz, +6.3dB, Q=0.8), PeakingEQ(45Hz, +0.7dB, Q=0.9)

### Redline (2009) — 1.29 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(26Hz, +3.7dB, Q=1.1), LowShelf(20Hz, +3.3dB, Q=1.1), LowShelf(17Hz, +3.0dB, Q=1.1), LowShelf(17Hz, +2.3dB, Q=1.0), LowShelf(9Hz, +2.0dB, Q=0.8), LowShelf(5Hz, +1.2dB, Q=0.6)
- **Catalogue (1 filters)**: LowShelf(10Hz, +8.0dB, Q=0.8)

### Frozen (2013) — 1.30 dB — by aron7awol
- **Predicted (5 filters)**: PeakingEQ(34Hz, +1.9dB, Q=1.2), LowShelf(20Hz, +3.5dB, Q=1.4), LowShelf(26Hz, +4.2dB, Q=1.6), LowShelf(27Hz, +2.8dB, Q=1.4), LowShelf(10Hz, +1.5dB, Q=0.4)
- **Catalogue (5 filters)**: PeakingEQ(50Hz, +0.5dB, Q=1.0), LowShelf(22Hz, +3.9dB, Q=0.9), LowShelf(22Hz, +3.9dB, Q=0.9), LowShelf(22Hz, +3.9dB, Q=0.9), LowShelf(10Hz, +0.9dB, Q=0.9)

### Invincible (2021) — 1.33 dB — by aron7awol
- **Predicted (5 filters)**: LowShelf(19Hz, +4.1dB, Q=1.1), LowShelf(22Hz, +4.7dB, Q=1.0), LowShelf(32Hz, +2.4dB, Q=1.4), PeakingEQ(24Hz, +0.7dB, Q=1.1), PeakingEQ(5Hz, -1.4dB, Q=0.5)
- **Catalogue (2 filters)**: LowShelf(10Hz, +1.7dB, Q=1.2), LowShelf(29Hz, +4.1dB, Q=0.8)

### Gravity (2013) — 1.36 dB — by aron7awol
- **Predicted (6 filters)**: PeakingEQ(34Hz, +2.3dB, Q=1.3), LowShelf(21Hz, +4.0dB, Q=1.3), LowShelf(25Hz, +3.9dB, Q=1.4), LowShelf(25Hz, +2.3dB, Q=1.2), LowShelf(12Hz, +1.0dB, Q=0.3), LowShelf(5Hz, -0.9dB, Q=0.6)
- **Catalogue (6 filters)**: PeakingEQ(41Hz, +1.0dB, Q=1.0), LowShelf(15Hz, +4.6dB, Q=1.0), LowShelf(15Hz, +4.6dB, Q=1.0), LowShelf(18Hz, +4.0dB, Q=0.9), LowShelf(18Hz, +4.0dB, Q=0.9), LowShelf(18Hz, +4.0dB, Q=0.9)

### El Camino: A Breaking Bad Movie (2019) — 1.36 dB — by aron7awol
- **Predicted (5 filters)**: LowShelf(26Hz, +3.0dB, Q=1.3), LowShelf(21Hz, +4.3dB, Q=1.1), LowShelf(18Hz, +3.7dB, Q=1.0), LowShelf(12Hz, +2.9dB, Q=0.6), LowShelf(8Hz, +1.8dB, Q=0.5)
- **Catalogue (5 filters)**: PeakingEQ(48Hz, +1.2dB, Q=1.0), LowShelf(28Hz, +3.9dB, Q=1.1), LowShelf(28Hz, +3.9dB, Q=1.1), LowShelf(15Hz, +2.9dB, Q=1.1), LowShelf(15Hz, +2.9dB, Q=1.1)

### The Lion King (1994) — 1.38 dB — by aron7awol
- **Predicted (5 filters)**: LowShelf(36Hz, +2.3dB, Q=1.3), LowShelf(26Hz, +4.0dB, Q=1.1), LowShelf(23Hz, +4.8dB, Q=0.8), LowShelf(18Hz, +3.6dB, Q=0.8), LowShelf(7Hz, +2.3dB, Q=0.4)
- **Catalogue (4 filters)**: PeakingEQ(45Hz, +1.5dB, Q=1.0), LowShelf(23Hz, +4.6dB, Q=1.1), LowShelf(23Hz, +4.6dB, Q=1.1), LowShelf(23Hz, +4.6dB, Q=1.1)

### Venom: The Last Dance (2024) — 1.40 dB — by kaelaria
- **Predicted (6 filters)**: LowShelf(26Hz, +3.4dB, Q=1.0), LowShelf(22Hz, +4.4dB, Q=1.1), LowShelf(21Hz, +3.7dB, Q=1.4), LowShelf(20Hz, +2.2dB, Q=1.4), LowShelf(11Hz, +1.0dB, Q=0.9), LowShelf(5Hz, +0.7dB, Q=0.5)
- **Catalogue (4 filters)**: LowShelf(22Hz, +7.8dB, Q=0.8), LowShelf(22Hz, +7.8dB, Q=0.8), PeakingEQ(31Hz, -2.6dB, Q=5.0), PeakingEQ(40Hz, -0.7dB, Q=3.0)

### Apocalypto (2006) — 1.41 dB — by aron7awol
- **Predicted (6 filters)**: PeakingEQ(42Hz, +1.9dB, Q=1.2), LowShelf(23Hz, +3.9dB, Q=1.5), LowShelf(32Hz, +4.2dB, Q=1.6), LowShelf(25Hz, +2.2dB, Q=1.3), LowShelf(7Hz, +0.7dB, Q=0.3), LowShelf(5Hz, -0.8dB, Q=0.8)
- **Catalogue (6 filters)**: PeakingEQ(65Hz, +0.2dB, Q=1.0), LowShelf(22Hz, +5.0dB, Q=0.8), LowShelf(22Hz, +5.0dB, Q=0.8), LowShelf(22Hz, +5.0dB, Q=0.8), LowShelf(10Hz, +3.7dB, Q=0.9), LowShelf(10Hz, +3.7dB, Q=0.9)

### The Wheel of Time (2021) — 1.49 dB — by aron7awol
- **Predicted (5 filters)**: LowShelf(21Hz, +3.9dB, Q=1.2), LowShelf(31Hz, +3.8dB, Q=1.2), LowShelf(26Hz, +3.1dB, Q=1.3), PeakingEQ(28Hz, +1.7dB, Q=1.3), PeakingEQ(5Hz, -1.0dB, Q=0.7)
- **Catalogue (4 filters)**: PeakingEQ(17Hz, -3.7dB, Q=6.0), LowShelf(22Hz, +6.4dB, Q=0.8), LowShelf(22Hz, +6.4dB, Q=0.8), PeakingEQ(70Hz, +0.2dB, Q=1.0)

### The Studio (2025) — 1.49 dB — by kaelaria
- **Predicted (6 filters)**: LowShelf(32Hz, +3.8dB, Q=1.2), LowShelf(26Hz, +3.4dB, Q=1.3), LowShelf(20Hz, +5.0dB, Q=1.1), LowShelf(19Hz, +4.3dB, Q=1.2), LowShelf(16Hz, +2.5dB, Q=1.2), PeakingEQ(14Hz, +1.5dB, Q=1.2)
- **Catalogue (5 filters)**: LowShelf(10Hz, +10.0dB, Q=0.8), LowShelf(20Hz, +8.0dB, Q=0.8), LowShelf(20Hz, +8.0dB, Q=0.8), LowShelf(20Hz, +8.0dB, Q=0.8), PeakingEQ(40Hz, -0.8dB, Q=3.0)

### Cowboy Bebop (1998) — 1.52 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(32Hz, +1.9dB, Q=1.6), LowShelf(27Hz, +2.3dB, Q=1.0), LowShelf(23Hz, +4.1dB, Q=1.0), LowShelf(24Hz, +4.1dB, Q=1.1), LowShelf(21Hz, +3.9dB, Q=1.0), LowShelf(10Hz, +2.9dB, Q=0.6)
- **Catalogue (4 filters)**: LowShelf(10Hz, +2.0dB, Q=1.0), LowShelf(22Hz, +6.0dB, Q=0.8), LowShelf(22Hz, +6.0dB, Q=0.8), LowShelf(22Hz, +6.0dB, Q=0.8)

### Requiem for a Dream (2000) — 1.53 dB — by aron7awol
- **Predicted (5 filters)**: LowShelf(25Hz, +2.9dB, Q=1.5), LowShelf(23Hz, +3.6dB, Q=1.3), LowShelf(20Hz, +3.6dB, Q=0.8), LowShelf(16Hz, +3.1dB, Q=0.7), LowShelf(11Hz, +1.8dB, Q=0.5)
- **Catalogue (3 filters)**: LowShelf(20Hz, +3.8dB, Q=0.8), LowShelf(20Hz, +3.8dB, Q=0.8), PeakingEQ(70Hz, +0.1dB, Q=1.0)

### 28 Weeks Later (2007) — 1.54 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(26Hz, +4.5dB, Q=1.2), LowShelf(20Hz, +3.7dB, Q=1.2), LowShelf(15Hz, +3.0dB, Q=1.1), LowShelf(15Hz, +2.2dB, Q=0.9), LowShelf(8Hz, +1.8dB, Q=0.7), LowShelf(5Hz, +1.2dB, Q=0.4)
- **Catalogue (2 filters)**: PeakingEQ(20Hz, +2.5dB, Q=1.5), LowShelf(20Hz, +10.6dB, Q=0.8)

### KPop Demon Hunters (2025) — 1.54 dB — by remixmark
- **Predicted (5 filters)**: LowShelf(17Hz, +2.9dB, Q=1.6), LowShelf(21Hz, +4.3dB, Q=1.6), LowShelf(26Hz, +6.3dB, Q=0.9), LowShelf(31Hz, +4.3dB, Q=1.6), PeakingEQ(27Hz, +1.6dB, Q=1.3)
- **Catalogue (6 filters)**: LowShelf(10Hz, +11.1dB, Q=0.9), LowShelf(21Hz, +6.4dB, Q=0.9), LowShelf(21Hz, +6.4dB, Q=0.9), LowShelf(21Hz, +6.4dB, Q=0.9), LowShelf(21Hz, +6.4dB, Q=0.9), PeakingEQ(50Hz, +1.2dB, Q=0.9)

### Rush Hour 3 (2007) — 1.57 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(31Hz, +4.0dB, Q=1.3), LowShelf(23Hz, +2.0dB, Q=1.6), LowShelf(18Hz, +3.1dB, Q=1.3), LowShelf(16Hz, +3.4dB, Q=0.9), LowShelf(11Hz, +3.1dB, Q=0.8), LowShelf(6Hz, +1.4dB, Q=0.6)
- **Catalogue (2 filters)**: LowShelf(20Hz, +12.0dB, Q=0.8), PeakingEQ(23Hz, +4.0dB, Q=3.0)

### Ice Age: Continental Drift (2012) — 1.57 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(26Hz, +2.6dB, Q=1.2), LowShelf(20Hz, +2.6dB, Q=1.0), LowShelf(18Hz, +3.6dB, Q=1.1), LowShelf(19Hz, +3.3dB, Q=1.1), LowShelf(14Hz, +3.2dB, Q=1.1), LowShelf(14Hz, +2.2dB, Q=1.2)
- **Catalogue (7 filters)**: LowShelf(10Hz, +3.0dB, Q=1.0), LowShelf(10Hz, +3.0dB, Q=1.0), PeakingEQ(12Hz, +2.0dB, Q=3.0), LowShelf(20Hz, +4.0dB, Q=0.8), LowShelf(20Hz, +4.0dB, Q=0.8), LowShelf(20Hz, +4.0dB, Q=0.8), LowShelf(20Hz, +4.0dB, Q=0.8)

### Ice Age: Dawn of the Dinosaurs (2009) — 1.58 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(25Hz, +2.5dB, Q=1.2), LowShelf(19Hz, +2.8dB, Q=1.0), LowShelf(18Hz, +3.8dB, Q=1.2), LowShelf(22Hz, +3.7dB, Q=1.2), LowShelf(16Hz, +3.5dB, Q=1.2), LowShelf(13Hz, +2.4dB, Q=1.4)
- **Catalogue (4 filters)**: LowShelf(11Hz, +3.0dB, Q=0.9), LowShelf(11Hz, +3.0dB, Q=0.9), PeakingEQ(20Hz, +1.0dB, Q=3.0), LowShelf(23Hz, +3.0dB, Q=0.8)

### Cosmos (2014) — 1.58 dB — by aron7awol
- **Predicted (6 filters)**: LowShelf(42Hz, +0.7dB, Q=1.3), LowShelf(24Hz, +2.5dB, Q=1.0), LowShelf(20Hz, +3.1dB, Q=0.8), LowShelf(17Hz, +2.7dB, Q=0.7), LowShelf(9Hz, +2.1dB, Q=0.4), LowShelf(6Hz, +1.4dB, Q=0.3)
- **Catalogue (4 filters)**: LowShelf(35Hz, +0.5dB, Q=1.0), LowShelf(14Hz, +4.2dB, Q=0.9), LowShelf(14Hz, +4.2dB, Q=0.9), LowShelf(14Hz, +4.2dB, Q=0.9)

### tick, tick...BOOM! (2021) — 1.61 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(20Hz, +3.1dB, Q=1.5), LowShelf(21Hz, +2.7dB, Q=1.4), LowShelf(22Hz, +5.1dB, Q=0.9), LowShelf(23Hz, +4.7dB, Q=0.9), LowShelf(18Hz, +3.3dB, Q=1.0), PeakingEQ(16Hz, +1.7dB, Q=1.1)
- **Catalogue (5 filters)**: LowShelf(10Hz, +7.0dB, Q=1.0), LowShelf(20Hz, +7.0dB, Q=0.8), LowShelf(20Hz, +7.0dB, Q=0.8), LowShelf(20Hz, +7.0dB, Q=0.8), LowShelf(20Hz, +7.0dB, Q=0.8)

### Taboo (2017) — 1.64 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(32Hz, +3.5dB, Q=1.1), LowShelf(23Hz, +2.1dB, Q=1.2), LowShelf(19Hz, +3.1dB, Q=1.3), LowShelf(16Hz, +3.8dB, Q=0.8), LowShelf(14Hz, +3.3dB, Q=1.0), LowShelf(8Hz, +1.7dB, Q=0.9)
- **Catalogue (6 filters)**: PeakingEQ(15Hz, +5.0dB, Q=2.0), LowShelf(20Hz, +4.5dB, Q=0.8), LowShelf(20Hz, +4.5dB, Q=0.8), LowShelf(20Hz, +4.5dB, Q=0.8), LowShelf(20Hz, +4.5dB, Q=0.8), LowShelf(20Hz, +4.5dB, Q=0.8)

### Rebel Moon - Part One: A Child of Fire (2023) — 1.65 dB — by t1g8rsfan
- **Predicted (6 filters)**: LowShelf(24Hz, +1.7dB, Q=1.3), LowShelf(21Hz, +3.3dB, Q=1.0), LowShelf(20Hz, +3.6dB, Q=1.4), LowShelf(27Hz, +3.1dB, Q=1.3), LowShelf(19Hz, +2.8dB, Q=0.7), LowShelf(14Hz, +1.9dB, Q=0.7)
- **Catalogue (3 filters)**: PeakingEQ(15Hz, -7.7dB, Q=10.0), LowShelf(23Hz, +6.0dB, Q=0.9), PeakingEQ(57Hz, +0.2dB, Q=1.1)

### A Knight of the Seven Kingdoms (2026) — 1.65 dB — by kaelaria
- **Predicted (5 filters)**: LowShelf(27Hz, +4.2dB, Q=0.9), LowShelf(22Hz, +4.8dB, Q=1.2), LowShelf(22Hz, +4.7dB, Q=1.0), LowShelf(23Hz, +3.2dB, Q=1.1), LowShelf(12Hz, +1.0dB, Q=1.0)
- **Catalogue (4 filters)**: LowShelf(10Hz, +10.0dB, Q=0.8), LowShelf(10Hz, +10.0dB, Q=0.8), LowShelf(30Hz, +8.0dB, Q=0.8), PeakingEQ(40Hz, -1.5dB, Q=3.0)

### Dark Matter (2024) — 1.68 dB — by kaelaria
- **Predicted (6 filters)**: LowShelf(33Hz, +3.7dB, Q=1.0), LowShelf(26Hz, +3.3dB, Q=1.3), LowShelf(23Hz, +3.9dB, Q=1.5), LowShelf(17Hz, +3.2dB, Q=1.3), LowShelf(14Hz, +2.1dB, Q=1.0), LowShelf(8Hz, +1.0dB, Q=0.9)
- **Catalogue (6 filters)**: LowShelf(10Hz, +7.0dB, Q=0.8), PeakingEQ(14Hz, -2.2dB, Q=3.3), PeakingEQ(24Hz, -2.0dB, Q=6.0), PeakingEQ(24Hz, +5.0dB, Q=3.0), LowShelf(25Hz, +5.0dB, Q=0.8), PeakingEQ(40Hz, -0.8dB, Q=3.0)

### Paprika (2006) — 1.69 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(26Hz, +3.8dB, Q=1.1), LowShelf(24Hz, +2.5dB, Q=1.5), LowShelf(21Hz, +3.5dB, Q=1.4), LowShelf(19Hz, +3.1dB, Q=1.2), LowShelf(11Hz, +2.4dB, Q=1.0), LowShelf(7Hz, +1.3dB, Q=0.8)
- **Catalogue (5 filters)**: PeakingEQ(16Hz, +2.0dB, Q=2.0), LowShelf(16Hz, +5.0dB, Q=0.9), LowShelf(16Hz, +5.0dB, Q=0.9), LowShelf(16Hz, +5.0dB, Q=0.9), PeakingEQ(23Hz, +7.0dB, Q=4.0)

### Ghost in the Shell (1995) — 1.69 dB — by aron7awol
- **Predicted (6 filters)**: LowShelf(41Hz, +2.0dB, Q=1.3), LowShelf(24Hz, +3.8dB, Q=1.3), LowShelf(28Hz, +3.5dB, Q=1.4), LowShelf(25Hz, +4.6dB, Q=1.0), LowShelf(23Hz, +4.1dB, Q=0.9), PeakingEQ(31Hz, +2.5dB, Q=1.0)
- **Catalogue (4 filters)**: LowShelf(33Hz, +4.6dB, Q=0.9), LowShelf(33Hz, +4.6dB, Q=0.9), LowShelf(33Hz, +4.6dB, Q=0.9), PeakingEQ(75Hz, +0.6dB, Q=1.0)

### The Rip (2026) — 1.74 dB — by halcyon888
- **Predicted (5 filters)**: LowShelf(18Hz, +4.3dB, Q=1.2), LowShelf(21Hz, +4.6dB, Q=1.1), PeakingEQ(35Hz, +2.9dB, Q=1.9), LowShelf(37Hz, +3.0dB, Q=1.4), LowShelf(17Hz, +1.3dB, Q=0.8)
- **Catalogue (5 filters)**: LowShelf(20Hz, +5.0dB, Q=0.8), LowShelf(20Hz, +5.0dB, Q=0.8), LowShelf(20Hz, +5.0dB, Q=0.8), PeakingEQ(29Hz, -1.3dB, Q=2.0), PeakingEQ(50Hz, +0.3dB, Q=1.0)

### The Platform 2 (2024) — 1.75 dB — by kaelaria
- **Predicted (6 filters)**: LowShelf(27Hz, +3.4dB, Q=1.1), LowShelf(23Hz, +3.5dB, Q=1.2), LowShelf(21Hz, +3.5dB, Q=1.3), LowShelf(20Hz, +3.1dB, Q=1.2), LowShelf(15Hz, +1.8dB, Q=1.2), LowShelf(10Hz, +1.0dB, Q=0.7)
- **Catalogue (3 filters)**: LowShelf(10Hz, +2.0dB, Q=0.8), LowShelf(20Hz, +7.0dB, Q=0.8), PeakingEQ(35Hz, -0.5dB, Q=3.0)

### Blink Twice (2024) — 1.80 dB — by kaelaria
- **Predicted (6 filters)**: LowShelf(32Hz, +3.5dB, Q=1.2), LowShelf(27Hz, +2.9dB, Q=1.4), LowShelf(22Hz, +4.5dB, Q=1.2), LowShelf(20Hz, +4.2dB, Q=1.2), LowShelf(18Hz, +1.8dB, Q=1.3), LowShelf(7Hz, +1.7dB, Q=0.5)
- **Catalogue (5 filters)**: LowShelf(10Hz, +10.0dB, Q=0.8), LowShelf(24Hz, +5.7dB, Q=0.8), LowShelf(24Hz, +5.7dB, Q=0.8), LowShelf(24Hz, +5.7dB, Q=0.8), PeakingEQ(40Hz, -1.6dB, Q=3.0)

### Hero (2002) — 1.82 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(21Hz, +4.9dB, Q=1.2), LowShelf(21Hz, +3.9dB, Q=1.2), LowShelf(16Hz, +4.0dB, Q=1.3), LowShelf(16Hz, +3.5dB, Q=0.9), LowShelf(17Hz, +2.9dB, Q=1.2), LowShelf(13Hz, +2.0dB, Q=1.5)
- **Catalogue (6 filters)**: PeakingEQ(11Hz, +9.0dB, Q=2.0), LowShelf(12Hz, +6.0dB, Q=0.8), LowShelf(12Hz, +6.0dB, Q=0.8), LowShelf(12Hz, +6.0dB, Q=0.8), LowShelf(12Hz, +6.0dB, Q=0.8), LowShelf(12Hz, +6.0dB, Q=0.8)

### Landman (2024) — 1.86 dB — by halcyon888
- **Predicted (6 filters)**: LowShelf(28Hz, +1.8dB, Q=1.2), LowShelf(23Hz, +3.2dB, Q=1.1), LowShelf(25Hz, +3.4dB, Q=1.2), LowShelf(27Hz, +2.7dB, Q=1.1), LowShelf(16Hz, +1.6dB, Q=1.2), PeakingEQ(18Hz, +1.2dB, Q=0.7)
- **Catalogue (3 filters)**: LowShelf(14Hz, +6.0dB, Q=0.8), LowShelf(32Hz, +6.5dB, Q=0.8), PeakingEQ(92Hz, +0.1dB, Q=0.8)

### Civil War (2024) — 1.87 dB — by t1g8rsfan
- **Predicted (6 filters)**: LowShelf(32Hz, +2.5dB, Q=1.3), LowShelf(25Hz, +2.5dB, Q=1.2), LowShelf(23Hz, +3.5dB, Q=1.2), LowShelf(22Hz, +4.5dB, Q=1.0), LowShelf(22Hz, +3.4dB, Q=1.0), LowShelf(15Hz, +2.2dB, Q=0.7)
- **Catalogue (6 filters)**: LowShelf(14Hz, +4.9dB, Q=0.9), LowShelf(14Hz, +4.9dB, Q=0.9), LowShelf(20Hz, +5.0dB, Q=0.7), LowShelf(20Hz, +5.0dB, Q=0.7), LowShelf(20Hz, +5.0dB, Q=0.7), LowShelf(20Hz, +5.0dB, Q=0.7)

### Luca (2021) — 1.89 dB — by aron7awol
- **Predicted (6 filters)**: LowShelf(30Hz, +2.7dB, Q=0.9), LowShelf(23Hz, +3.9dB, Q=0.8), LowShelf(26Hz, +3.0dB, Q=0.9), LowShelf(20Hz, +1.8dB, Q=0.7), LowShelf(8Hz, +1.2dB, Q=0.3), LowShelf(5Hz, +0.7dB, Q=0.2)
- **Catalogue (6 filters)**: PeakingEQ(11Hz, -12.0dB, Q=4.0), PeakingEQ(13Hz, -6.0dB, Q=4.5), PeakingEQ(15Hz, -12.0dB, Q=5.0), LowShelf(15Hz, +6.9dB, Q=0.7), LowShelf(15Hz, +6.9dB, Q=0.7), LowShelf(15Hz, +6.9dB, Q=0.7)

### Reacher (2022) — 1.90 dB — by aron7awol
- **Predicted (5 filters)**: LowShelf(27Hz, +3.7dB, Q=1.5), LowShelf(23Hz, +4.5dB, Q=1.3), PeakingEQ(36Hz, +2.2dB, Q=1.4), LowShelf(18Hz, +1.9dB, Q=0.9), PeakingEQ(16Hz, +0.5dB, Q=0.9)
- **Catalogue (3 filters)**: LowShelf(31Hz, +5.8dB, Q=0.8), LowShelf(31Hz, +5.8dB, Q=0.8), PeakingEQ(100Hz, +0.1dB, Q=1.0)

### Inside Out 2 (2024) — 1.90 dB — by kaelaria
- **Predicted (5 filters)**: LowShelf(16Hz, +4.6dB, Q=1.2), LowShelf(19Hz, +5.3dB, Q=1.3), LowShelf(27Hz, +4.9dB, Q=1.8), PeakingEQ(30Hz, +2.4dB, Q=2.0), LowShelf(14Hz, +0.8dB, Q=0.7)
- **Catalogue (4 filters)**: LowShelf(17Hz, +7.5dB, Q=0.8), LowShelf(17Hz, +7.5dB, Q=0.8), PeakingEQ(21Hz, -0.5dB, Q=4.0), PeakingEQ(30Hz, -1.0dB, Q=3.0)

### Fallout (2024) — 1.91 dB — by kaelaria
- **Predicted (6 filters)**: LowShelf(26Hz, +3.6dB, Q=1.0), LowShelf(22Hz, +4.1dB, Q=1.2), LowShelf(21Hz, +4.0dB, Q=1.2), LowShelf(20Hz, +2.3dB, Q=1.3), LowShelf(11Hz, +1.4dB, Q=0.9), LowShelf(7Hz, +0.7dB, Q=0.6)
- **Catalogue (7 filters)**: LowShelf(10Hz, +5.3dB, Q=0.8), PeakingEQ(12Hz, -2.0dB, Q=5.0), PeakingEQ(15Hz, -1.2dB, Q=5.0), LowShelf(18Hz, +4.7dB, Q=0.8), LowShelf(18Hz, +4.7dB, Q=0.8), PeakingEQ(24Hz, -0.5dB, Q=8.0), PeakingEQ(25Hz, -1.5dB, Q=3.0)

### A Working Man (2025) — 1.92 dB — by kaelaria
- **Predicted (6 filters)**: LowShelf(26Hz, +2.9dB, Q=1.1), LowShelf(23Hz, +3.7dB, Q=1.2), LowShelf(21Hz, +4.2dB, Q=1.3), LowShelf(21Hz, +3.0dB, Q=1.3), LowShelf(13Hz, +1.5dB, Q=1.2), LowShelf(9Hz, +1.0dB, Q=0.7)
- **Catalogue (5 filters)**: PeakingEQ(14Hz, +5.0dB, Q=2.7), LowShelf(14Hz, +8.0dB, Q=0.8), LowShelf(14Hz, +8.0dB, Q=0.8), LowShelf(14Hz, +8.0dB, Q=0.8), PeakingEQ(20Hz, -1.0dB, Q=3.0)

### Black Mirror (2011) — 1.92 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(33Hz, +3.2dB, Q=1.2), LowShelf(20Hz, +2.6dB, Q=1.0), LowShelf(21Hz, +3.5dB, Q=0.9), LowShelf(23Hz, +4.3dB, Q=1.0), LowShelf(24Hz, +4.0dB, Q=1.1), LowShelf(18Hz, +2.5dB, Q=0.9)
- **Catalogue (5 filters)**: PeakingEQ(10Hz, +7.0dB, Q=1.7), PeakingEQ(13Hz, +2.0dB, Q=3.0), PeakingEQ(15Hz, +5.5dB, Q=1.9), LowShelf(20Hz, +7.0dB, Q=0.8), PeakingEQ(23Hz, +2.0dB, Q=4.0)

### 1923 (2022) — 1.93 dB — by kaelaria
- **Predicted (6 filters)**: LowShelf(24Hz, +4.3dB, Q=1.1), LowShelf(22Hz, +5.5dB, Q=1.0), LowShelf(23Hz, +4.7dB, Q=1.2), LowShelf(22Hz, +3.1dB, Q=1.3), LowShelf(15Hz, +1.1dB, Q=1.0), PeakingEQ(16Hz, +0.9dB, Q=0.6)
- **Catalogue (4 filters)**: LowShelf(20Hz, +0.0dB, Q=0.8), LowShelf(25Hz, +8.5dB, Q=0.8), LowShelf(25Hz, +8.5dB, Q=0.8), PeakingEQ(37Hz, -2.6dB, Q=3.0)

### The Lord of the Rings: The War of the Rohirrim (2024) — 1.93 dB — by t1g8rsfan
- **Predicted (5 filters)**: LowShelf(19Hz, +2.8dB, Q=1.1), LowShelf(20Hz, +4.1dB, Q=1.3), LowShelf(23Hz, +3.7dB, Q=1.5), LowShelf(26Hz, +3.2dB, Q=1.6), LowShelf(22Hz, +1.5dB, Q=0.7)
- **Catalogue (5 filters)**: LowShelf(12Hz, +3.0dB, Q=0.8), LowShelf(21Hz, +4.0dB, Q=1.0), LowShelf(21Hz, +4.0dB, Q=1.0), PeakingEQ(32Hz, -3.5dB, Q=10.0), PeakingEQ(36Hz, +0.8dB, Q=0.8)

### They Shall Not Grow Old (2018) — 1.93 dB — by mobe1969
- **Predicted (5 filters)**: LowShelf(16Hz, +5.3dB, Q=1.3), LowShelf(18Hz, +4.0dB, Q=1.4), LowShelf(22Hz, +3.3dB, Q=1.6), LowShelf(24Hz, +2.3dB, Q=1.4), LowShelf(11Hz, +1.4dB, Q=0.7)
- **Catalogue (6 filters)**: LowShelf(15Hz, +3.4dB, Q=0.9), LowShelf(15Hz, +3.4dB, Q=0.9), LowShelf(15Hz, +3.4dB, Q=0.9), LowShelf(15Hz, +3.4dB, Q=0.9), PeakingEQ(16Hz, +4.0dB, Q=3.0), PeakingEQ(23Hz, +3.0dB, Q=5.0)

### Ghost in the Shell 2.0 (2008) — 1.95 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(16Hz, +4.7dB, Q=1.2), LowShelf(18Hz, +3.9dB, Q=1.2), LowShelf(26Hz, +3.3dB, Q=1.7), LowShelf(27Hz, +2.0dB, Q=1.5), LowShelf(11Hz, +1.0dB, Q=0.7), PeakingEQ(5Hz, -0.6dB, Q=1.1)
- **Catalogue (5 filters)**: PeakingEQ(13Hz, +2.0dB, Q=3.0), LowShelf(13Hz, +3.7dB, Q=0.8), LowShelf(13Hz, +3.7dB, Q=0.8), LowShelf(13Hz, +3.7dB, Q=0.8), PeakingEQ(22Hz, +6.0dB, Q=3.0)

### Ice Age: The Meltdown (2006) — 1.96 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(25Hz, +2.9dB, Q=1.1), LowShelf(19Hz, +2.8dB, Q=1.0), LowShelf(18Hz, +3.6dB, Q=1.1), LowShelf(20Hz, +3.4dB, Q=1.1), LowShelf(15Hz, +3.3dB, Q=1.1), LowShelf(12Hz, +2.3dB, Q=1.3)
- **Catalogue (6 filters)**: LowShelf(10Hz, +6.0dB, Q=1.0), LowShelf(17Hz, +5.5dB, Q=0.8), LowShelf(17Hz, +5.5dB, Q=0.8), LowShelf(17Hz, +5.5dB, Q=0.8), LowShelf(17Hz, +6.0dB, Q=1.0), PeakingEQ(20Hz, +5.0dB, Q=1.7)

### Tangled (2010) — 1.96 dB — by aron7awol
- **Predicted (5 filters)**: LowShelf(33Hz, +2.4dB, Q=1.3), LowShelf(21Hz, +4.3dB, Q=1.3), LowShelf(28Hz, +3.4dB, Q=1.6), LowShelf(29Hz, +2.8dB, Q=1.3), LowShelf(11Hz, +1.7dB, Q=0.5)
- **Catalogue (6 filters)**: PeakingEQ(62Hz, +1.3dB, Q=1.0), LowShelf(31Hz, +7.0dB, Q=0.9), LowShelf(31Hz, +7.0dB, Q=0.9), LowShelf(16Hz, +7.0dB, Q=1.1), LowShelf(16Hz, +7.0dB, Q=1.1), LowShelf(16Hz, +7.0dB, Q=1.1)

### Baby Reindeer (2024) — 1.97 dB — by kaelaria
- **Predicted (6 filters)**: LowShelf(27Hz, +3.3dB, Q=1.2), LowShelf(22Hz, +4.5dB, Q=1.0), LowShelf(20Hz, +4.4dB, Q=1.2), LowShelf(22Hz, +2.4dB, Q=1.7), LowShelf(15Hz, +1.8dB, Q=0.9), LowShelf(10Hz, +1.5dB, Q=0.6)
- **Catalogue (4 filters)**: LowShelf(18Hz, +7.0dB, Q=0.8), LowShelf(18Hz, +7.0dB, Q=0.8), LowShelf(18Hz, +7.0dB, Q=0.8), PeakingEQ(23Hz, -4.6dB, Q=2.5)

### Scavengers Reign (2023) — 1.97 dB — by halcyon888
- **Predicted (6 filters)**: LowShelf(26Hz, +1.1dB, Q=1.3), LowShelf(22Hz, +2.6dB, Q=1.0), LowShelf(24Hz, +2.9dB, Q=1.1), LowShelf(23Hz, +2.3dB, Q=1.1), LowShelf(15Hz, +1.9dB, Q=1.0), LowShelf(16Hz, +1.1dB, Q=0.9)
- **Catalogue (4 filters)**: PeakingEQ(22Hz, +1.0dB, Q=3.0), LowShelf(22Hz, +5.0dB, Q=0.8), LowShelf(22Hz, +5.0dB, Q=0.8), PeakingEQ(24Hz, +4.0dB, Q=3.0)

### Cloudy with a Chance of Meatballs (2009) — 1.98 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(13Hz, +4.6dB, Q=1.2), LowShelf(19Hz, +4.4dB, Q=1.1), LowShelf(23Hz, +4.3dB, Q=1.5), LowShelf(26Hz, +3.8dB, Q=1.4), LowShelf(18Hz, +3.0dB, Q=1.2), PeakingEQ(14Hz, +0.6dB, Q=2.1)
- **Catalogue (2 filters)**: LowShelf(18Hz, +11.0dB, Q=0.9), PeakingEQ(22Hz, +5.0dB, Q=3.5)

## Good predictions — 2 to 4 dB (104 titles)

Usable as a starting point. The predicted filters are in the right
ballpark but may need manual tweaking of gain or frequency.

### Before I Go to Sleep (2014) — 2.02 dB — by mobe1969
- **Predicted (5 filters)**: LowShelf(16Hz, +5.4dB, Q=1.3), LowShelf(18Hz, +4.4dB, Q=1.3), LowShelf(23Hz, +3.6dB, Q=1.6), LowShelf(25Hz, +2.1dB, Q=1.5), LowShelf(11Hz, +1.1dB, Q=0.7)
- **Catalogue (1 filters)**: LowShelf(12Hz, +14.0dB, Q=0.8)

### Our Planet (2019) — 2.04 dB — by kaelaria
- **Predicted (6 filters)**: LowShelf(29Hz, +3.3dB, Q=0.9), LowShelf(22Hz, +3.5dB, Q=1.2), LowShelf(21Hz, +3.3dB, Q=1.4), LowShelf(20Hz, +1.7dB, Q=1.1), LowShelf(7Hz, +1.0dB, Q=0.5), LowShelf(5Hz, +0.7dB, Q=0.3)
- **Catalogue (5 filters)**: PeakingEQ(13Hz, -5.0dB, Q=1.0), LowShelf(21Hz, +6.5dB, Q=0.7), LowShelf(21Hz, +6.5dB, Q=0.7), PeakingEQ(29Hz, +3.6dB, Q=2.0), PeakingEQ(45Hz, -1.0dB, Q=4.0)

### Wyrmwood (2014) — 2.06 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(16Hz, +5.0dB, Q=1.3), LowShelf(19Hz, +4.4dB, Q=1.4), LowShelf(24Hz, +3.4dB, Q=1.6), LowShelf(24Hz, +2.1dB, Q=1.3), LowShelf(10Hz, +0.9dB, Q=0.5), PeakingEQ(5Hz, -0.6dB, Q=0.9)
- **Catalogue (4 filters)**: LowShelf(20Hz, +1.0dB, Q=0.8), LowShelf(20Hz, +1.0dB, Q=0.8), LowShelf(20Hz, +1.0dB, Q=0.8), LowShelf(20Hz, +1.0dB, Q=0.8)

### The Expanse (2015) — 2.07 dB — by aron7awol
- **Predicted (6 filters)**: LowShelf(36Hz, +1.8dB, Q=1.3), LowShelf(22Hz, +3.6dB, Q=1.3), LowShelf(23Hz, +3.2dB, Q=1.3), LowShelf(20Hz, +2.1dB, Q=1.1), LowShelf(12Hz, +0.8dB, Q=0.3), LowShelf(5Hz, -0.8dB, Q=0.6)
- **Catalogue (6 filters)**: PeakingEQ(100Hz, +0.3dB, Q=1.0), LowShelf(10Hz, -4.4dB, Q=0.9), LowShelf(10Hz, -4.4dB, Q=0.9), LowShelf(21Hz, +4.7dB, Q=0.9), LowShelf(21Hz, +4.7dB, Q=0.9), LowShelf(60Hz, +2.0dB, Q=1.1)

### Undone (2019) — 2.13 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(19Hz, +3.4dB, Q=1.3), LowShelf(20Hz, +4.1dB, Q=1.0), LowShelf(20Hz, +4.3dB, Q=1.0), LowShelf(20Hz, +4.1dB, Q=1.2), LowShelf(19Hz, +2.2dB, Q=1.2), PeakingEQ(16Hz, +1.0dB, Q=0.8)
- **Catalogue (5 filters)**: LowShelf(11Hz, +3.0dB, Q=0.8), LowShelf(11Hz, +3.0dB, Q=0.8), LowShelf(11Hz, +3.0dB, Q=0.8), LowShelf(11Hz, +3.0dB, Q=0.8), LowShelf(35Hz, +4.0dB, Q=0.8)

### Edge of Tomorrow (2014) — 2.14 dB — by aron7awol
- **Predicted (6 filters)**: PeakingEQ(43Hz, +1.9dB, Q=1.2), LowShelf(21Hz, +3.3dB, Q=1.7), LowShelf(30Hz, +3.9dB, Q=1.7), LowShelf(25Hz, +2.7dB, Q=1.4), LowShelf(9Hz, +1.3dB, Q=0.3), LowShelf(5Hz, -0.7dB, Q=0.6)
- **Catalogue (5 filters)**: LowShelf(23Hz, +6.9dB, Q=0.9), LowShelf(23Hz, +6.9dB, Q=0.9), LowShelf(23Hz, +6.9dB, Q=0.9), LowShelf(23Hz, +6.9dB, Q=0.9), PeakingEQ(54Hz, +1.1dB, Q=0.9)

### Slow Horses (2022) — 2.15 dB — by kaelaria
- **Predicted (6 filters)**: LowShelf(25Hz, +5.2dB, Q=1.0), LowShelf(24Hz, +5.9dB, Q=1.3), LowShelf(24Hz, +4.9dB, Q=1.2), LowShelf(24Hz, +2.6dB, Q=1.3), PeakingEQ(16Hz, +1.6dB, Q=1.6), PeakingEQ(12Hz, +1.4dB, Q=1.1)
- **Catalogue (4 filters)**: LowShelf(20Hz, +8.0dB, Q=0.8), LowShelf(20Hz, +8.0dB, Q=0.8), LowShelf(20Hz, +8.0dB, Q=0.8), PeakingEQ(35Hz, -1.7dB, Q=3.0)

### Sugar (2024) — 2.15 dB — by kaelaria
- **Predicted (6 filters)**: LowShelf(27Hz, +3.5dB, Q=1.1), LowShelf(25Hz, +3.5dB, Q=1.3), LowShelf(22Hz, +4.4dB, Q=1.4), LowShelf(20Hz, +3.1dB, Q=1.2), LowShelf(12Hz, +1.9dB, Q=0.9), LowShelf(7Hz, +0.7dB, Q=0.6)
- **Catalogue (3 filters)**: LowShelf(25Hz, +7.6dB, Q=0.8), LowShelf(25Hz, +7.6dB, Q=0.8), PeakingEQ(36Hz, -2.3dB, Q=3.0)

### Blue Planet II (2017) — 2.16 dB — by halcyon888
- **Predicted (6 filters)**: LowShelf(19Hz, +2.1dB, Q=1.3), LowShelf(20Hz, +3.0dB, Q=1.4), LowShelf(28Hz, +2.9dB, Q=1.7), LowShelf(23Hz, +2.7dB, Q=1.3), LowShelf(20Hz, +0.7dB, Q=1.1), PeakingEQ(14Hz, -0.8dB, Q=0.9)
- **Catalogue (2 filters)**: LowShelf(10Hz, -1.3dB, Q=0.9), LowShelf(24Hz, +3.8dB, Q=0.8)

### The Secret World of Arrietty (2010) — 2.17 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(21Hz, +3.8dB, Q=1.4), LowShelf(20Hz, +3.4dB, Q=1.3), LowShelf(21Hz, +4.2dB, Q=1.1), LowShelf(22Hz, +3.5dB, Q=1.1), PeakingEQ(19Hz, +3.4dB, Q=1.4), PeakingEQ(16Hz, +2.4dB, Q=1.5)
- **Catalogue (4 filters)**: LowShelf(10Hz, +4.0dB, Q=0.8), LowShelf(10Hz, +4.0dB, Q=0.8), LowShelf(30Hz, +3.0dB, Q=0.8), LowShelf(30Hz, +3.0dB, Q=0.8)

### Sicario (2015) — 2.22 dB — by aron7awol
- **Predicted (5 filters)**: LowShelf(24Hz, +2.5dB, Q=1.3), LowShelf(19Hz, +3.9dB, Q=1.0), LowShelf(18Hz, +3.2dB, Q=0.9), LowShelf(10Hz, +2.4dB, Q=0.7), LowShelf(6Hz, +1.5dB, Q=0.5)
- **Catalogue (6 filters)**: PeakingEQ(20Hz, +3.0dB, Q=6.0), LowShelf(20Hz, -4.3dB, Q=0.9), LowShelf(20Hz, -4.3dB, Q=0.9), LowShelf(20Hz, -4.3dB, Q=0.9), LowShelf(27Hz, +6.4dB, Q=0.9), LowShelf(27Hz, +6.4dB, Q=0.9)

### The Wandering Earth (2019) — 2.23 dB — by aron7awol
- **Predicted (5 filters)**: PeakingEQ(41Hz, +1.6dB, Q=1.2), LowShelf(23Hz, +4.0dB, Q=1.5), LowShelf(28Hz, +4.4dB, Q=1.6), LowShelf(28Hz, +3.1dB, Q=1.4), LowShelf(9Hz, +1.2dB, Q=0.3)
- **Catalogue (5 filters)**: PeakingEQ(37Hz, +2.8dB, Q=1.0), PeakingEQ(10Hz, -3.0dB, Q=0.5), LowShelf(10Hz, +5.0dB, Q=0.9), LowShelf(22Hz, +6.5dB, Q=1.2), LowShelf(22Hz, +6.5dB, Q=1.2)

### Riders of Justice (2020) — 2.23 dB — by mobe1969
- **Predicted (5 filters)**: LowShelf(16Hz, +4.5dB, Q=1.3), LowShelf(18Hz, +4.3dB, Q=1.2), LowShelf(25Hz, +4.0dB, Q=1.6), LowShelf(27Hz, +3.1dB, Q=1.5), LowShelf(14Hz, +2.1dB, Q=0.8)
- **Catalogue (5 filters)**: LowShelf(10Hz, +6.4dB, Q=0.8), LowShelf(19Hz, +6.7dB, Q=0.8), LowShelf(19Hz, +6.7dB, Q=0.8), LowShelf(19Hz, +6.7dB, Q=0.8), LowShelf(19Hz, +6.7dB, Q=0.8)

### Prisoners (2013) — 2.29 dB — by mobe1969
- **Predicted (5 filters)**: LowShelf(17Hz, +5.1dB, Q=1.4), LowShelf(18Hz, +4.3dB, Q=1.2), LowShelf(25Hz, +3.6dB, Q=1.6), LowShelf(26Hz, +2.3dB, Q=1.5), LowShelf(12Hz, +1.2dB, Q=0.8)
- **Catalogue (6 filters)**: LowShelf(10Hz, +6.4dB, Q=0.9), LowShelf(10Hz, +6.4dB, Q=0.9), LowShelf(20Hz, +5.4dB, Q=0.8), LowShelf(20Hz, +5.4dB, Q=0.8), LowShelf(20Hz, +5.4dB, Q=0.8), LowShelf(20Hz, +5.4dB, Q=0.8)

### The Lego Movie (2014) — 2.30 dB — by aron7awol
- **Predicted (5 filters)**: LowShelf(37Hz, +1.8dB, Q=1.3), LowShelf(22Hz, +3.5dB, Q=1.5), LowShelf(25Hz, +3.7dB, Q=1.5), LowShelf(23Hz, +3.4dB, Q=1.3), LowShelf(14Hz, +1.6dB, Q=0.6)
- **Catalogue (4 filters)**: PeakingEQ(40Hz, +0.7dB, Q=1.0), LowShelf(17Hz, +5.7dB, Q=0.9), LowShelf(17Hz, +5.7dB, Q=0.9), LowShelf(17Hz, +5.7dB, Q=0.9)

*... and 89 more titles in this range.*

## Needs work — over 5 dB (35 titles)

These predictions are too far from the catalogue to be directly usable.
Typically older films (pre-1990) or titles with unusual rolloff patterns.

### Metropolis (2001) — 5.02 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(23Hz, +4.0dB, Q=1.4), LowShelf(21Hz, +4.5dB, Q=1.3), LowShelf(20Hz, +4.8dB, Q=1.3), LowShelf(19Hz, +4.4dB, Q=1.1), LowShelf(17Hz, +3.1dB, Q=1.3), LowShelf(15Hz, +2.2dB, Q=1.2)
- **Catalogue (4 filters)**: LowShelf(10Hz, +5.0dB, Q=0.8), LowShelf(27Hz, +5.0dB, Q=0.8), LowShelf(27Hz, +5.0dB, Q=0.8), LowShelf(27Hz, +5.0dB, Q=0.8)

### Don't Breathe (2016) — 5.03 dB — by aron7awol
- **Predicted (5 filters)**: LowShelf(29Hz, +2.4dB, Q=1.4), LowShelf(20Hz, +3.5dB, Q=1.6), LowShelf(28Hz, +4.4dB, Q=1.6), LowShelf(28Hz, +3.3dB, Q=1.5), LowShelf(18Hz, +1.7dB, Q=0.5)
- **Catalogue (3 filters)**: LowShelf(40Hz, -4.0dB, Q=0.8), LowShelf(16Hz, +3.8dB, Q=1.1), LowShelf(16Hz, +3.8dB, Q=1.1)

### The Abyss (1989) — 5.10 dB — by mobe1969
- **Predicted (5 filters)**: LowShelf(20Hz, +5.6dB, Q=1.4), LowShelf(21Hz, +6.0dB, Q=1.3), LowShelf(27Hz, +5.4dB, Q=1.7), LowShelf(28Hz, +3.7dB, Q=2.1), LowShelf(16Hz, +2.9dB, Q=1.1)
- **Catalogue (6 filters)**: LowShelf(10Hz, +5.0dB, Q=0.9), LowShelf(10Hz, +5.0dB, Q=0.9), LowShelf(26Hz, +6.0dB, Q=0.8), LowShelf(26Hz, +6.0dB, Q=0.8), LowShelf(26Hz, +6.0dB, Q=0.8), LowShelf(26Hz, +6.0dB, Q=0.8)

### Blade Runner (1982) — 5.11 dB — by aron7awol
- **Predicted (6 filters)**: PeakingEQ(51Hz, +3.9dB, Q=1.2), LowShelf(27Hz, +4.6dB, Q=1.5), LowShelf(27Hz, +5.7dB, Q=1.4), LowShelf(25Hz, +5.2dB, Q=0.7), LowShelf(23Hz, +5.1dB, Q=0.7), LowShelf(25Hz, +2.2dB, Q=0.8)
- **Catalogue (5 filters)**: PeakingEQ(40Hz, +0.8dB, Q=1.0), LowShelf(17Hz, +4.8dB, Q=0.9), LowShelf(17Hz, +4.8dB, Q=0.9), LowShelf(17Hz, +4.8dB, Q=0.9), LowShelf(17Hz, +4.8dB, Q=0.9)

### Chronicle (2012) — 5.13 dB — by mobe1969
- **Predicted (5 filters)**: LowShelf(18Hz, +5.0dB, Q=1.3), LowShelf(18Hz, +4.5dB, Q=1.2), LowShelf(20Hz, +3.3dB, Q=1.4), LowShelf(20Hz, +2.6dB, Q=1.3), LowShelf(13Hz, +1.7dB, Q=0.9)
- **Catalogue (10 filters)**: LowShelf(10Hz, +5.4dB, Q=1.0), LowShelf(10Hz, +5.4dB, Q=1.0), PeakingEQ(15Hz, +2.7dB, Q=3.0), LowShelf(19Hz, +6.0dB, Q=0.8), LowShelf(19Hz, +6.0dB, Q=0.8), LowShelf(19Hz, +6.0dB, Q=0.8), LowShelf(19Hz, +6.0dB, Q=0.8), PeakingEQ(20Hz, +5.4dB, Q=2.0), PeakingEQ(24Hz, +4.0dB, Q=5.0), PeakingEQ(35Hz, +2.0dB, Q=8.0)

### Ghost in the Shell: Stand Alone Complex - The Laughing Man (2005) — 5.19 dB — by mobe1969
- **Predicted (5 filters)**: LowShelf(14Hz, +5.1dB, Q=1.6), LowShelf(20Hz, +5.1dB, Q=1.3), LowShelf(24Hz, +3.5dB, Q=1.9), LowShelf(22Hz, +3.0dB, Q=1.0), LowShelf(17Hz, +2.2dB, Q=0.9)
- **Catalogue (6 filters)**: LowShelf(10Hz, +3.0dB, Q=1.0), LowShelf(10Hz, +3.0dB, Q=1.0), LowShelf(20Hz, +4.0dB, Q=1.0), LowShelf(36Hz, +3.4dB, Q=0.8), LowShelf(36Hz, +3.4dB, Q=0.8), LowShelf(36Hz, +3.4dB, Q=0.8)

### Your Name. (2016) — 5.30 dB — by mobe1969
- **Predicted (5 filters)**: LowShelf(15Hz, +4.3dB, Q=1.3), LowShelf(19Hz, +4.2dB, Q=1.1), LowShelf(24Hz, +2.9dB, Q=1.5), LowShelf(23Hz, +2.7dB, Q=1.1), LowShelf(14Hz, +2.2dB, Q=1.0)
- **Catalogue (4 filters)**: LowShelf(31Hz, +3.4dB, Q=0.8), LowShelf(31Hz, +3.4dB, Q=0.8), LowShelf(31Hz, +3.4dB, Q=0.8), PeakingEQ(33Hz, +5.0dB, Q=3.0)

### The Night Manager (2016) — 5.32 dB — by mobe1969
- **Predicted (5 filters)**: LowShelf(15Hz, +3.8dB, Q=1.2), LowShelf(18Hz, +3.9dB, Q=1.2), LowShelf(26Hz, +3.9dB, Q=1.6), LowShelf(30Hz, +3.1dB, Q=1.6), LowShelf(15Hz, +1.9dB, Q=0.8)
- **Catalogue (6 filters)**: PeakingEQ(12Hz, +4.0dB, Q=3.0), LowShelf(35Hz, +3.0dB, Q=0.8), LowShelf(35Hz, +3.0dB, Q=0.8), LowShelf(35Hz, +3.0dB, Q=0.8), PeakingEQ(36Hz, +13.0dB, Q=12.0), PeakingEQ(45Hz, +7.0dB, Q=14.0)

### Dream Scenario (2023) — 5.53 dB — by kaelaria
- **Predicted (5 filters)**: LowShelf(17Hz, +4.4dB, Q=1.2), LowShelf(22Hz, +5.4dB, Q=1.3), LowShelf(30Hz, +5.4dB, Q=1.8), LowShelf(34Hz, +3.0dB, Q=1.9), PeakingEQ(9Hz, -0.8dB, Q=1.2)
- **Catalogue (3 filters)**: LowShelf(12Hz, +5.4dB, Q=0.8), LowShelf(12Hz, +5.4dB, Q=0.8), PeakingEQ(16Hz, -2.0dB, Q=3.0)

### Street Fighter (1994) — 5.54 dB — by mobe1969
- **Predicted (6 filters)**: LowShelf(15Hz, +4.8dB, Q=1.5), LowShelf(18Hz, +5.6dB, Q=1.4), LowShelf(19Hz, +4.7dB, Q=1.2), LowShelf(20Hz, +4.6dB, Q=1.1), LowShelf(20Hz, +3.7dB, Q=1.2), LowShelf(14Hz, +2.8dB, Q=1.0)
- **Catalogue (7 filters)**: LowShelf(10Hz, +4.0dB, Q=1.0), LowShelf(10Hz, +4.0dB, Q=1.0), LowShelf(10Hz, +4.0dB, Q=1.0), PeakingEQ(17Hz, +2.5dB, Q=3.0), PeakingEQ(26Hz, +2.7dB, Q=1.4), PeakingEQ(38Hz, +1.6dB, Q=4.0), LowShelf(38Hz, +4.7dB, Q=0.9)

*... and 25 more titles in this range.*

