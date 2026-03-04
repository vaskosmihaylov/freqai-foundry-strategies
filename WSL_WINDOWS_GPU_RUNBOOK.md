# WSL + Docker GPU Runbook (QuickAdapter + ReforceXY)

This runbook is for running the strategies in this repository on:
- Windows 10 + WSL2
- Docker Desktop (WSL backend)
- NVIDIA RTX 4080 Super
- AMD Ryzen 7 7800X3D

## 1. Preconditions (Windows + WSL)

1. Install/update NVIDIA Windows driver (Game Ready or Studio).
2. Enable WSL2.
3. Install Docker Desktop and enable:
   - `Use the WSL 2 based engine`
   - WSL integration for your distro (`Settings -> Resources -> WSL Integration`)
   - GPU support (NVIDIA)
4. Start Docker Desktop.

## 2. Verify GPU from WSL

Run these in your WSL terminal:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Both commands should show your RTX 4080 Super.

## 3. QuickAdapter (GPU-enabled)

The repository already contains:
- `quickadapter/docker-compose.windows-gpu.yml`
- `quickadapter/user_data/config.windows-gpu.json`

### 3.1 Prepare config

```bash
cd /Users/vasmihay/Personal/AI_Projects/freqai-foundry-strategies/quickadapter
cp -n user_data/config-template.json user_data/config.json
```

Edit `user_data/config.json` and set:
- `exchange.key` / `exchange.secret`
- `exchange.pair_whitelist`
- `trading_mode` (`spot` or `futures`)
- optional risk values (`max_open_trades`, leverage-related fields for futures)

### 3.2 Build and start

```bash
docker compose -f docker-compose.yml -f docker-compose.windows-gpu.yml up -d --build
docker compose ps
docker compose logs -f freqtrade
```

### 3.3 One-off run command (recommended pattern)

Always combine base config + GPU override config:

```bash
docker compose -f docker-compose.yml -f docker-compose.windows-gpu.yml run --rm freqtrade trade \
  --config user_data/config.json \
  --config user_data/config.windows-gpu.json \
  --strategy-path user_data/strategies \
  --strategy QuickAdapterV3
```

### 3.4 Backtesting example

```bash
docker compose -f docker-compose.yml -f docker-compose.windows-gpu.yml run --rm freqtrade backtesting \
  --config user_data/config.json \
  --config user_data/config.windows-gpu.json \
  --strategy-path user_data/strategies \
  --strategy QuickAdapterV3 \
  --timerange 20250101-20250201 \
  --timeframe 5m \
  --export trades
```

## 4. ReforceXY (using the same GPU compose override)

`ReforceXY` does not currently include its own `docker-compose.windows-gpu.yml`.
Use QuickAdapter's GPU compose override file from the parent path.

### 4.1 Prepare config

```bash
cd /Users/vasmihay/Personal/AI_Projects/freqai-foundry-strategies/ReforceXY
cp -n user_data/config-template.json user_data/config.json
```

In `user_data/config.json`, set:
- exchange credentials and pairs
- `freqai.model_training_parameters.device` to `"cuda"` (template default is `"auto"`)

### 4.2 Build and start

```bash
docker compose -f docker-compose.yml -f ../quickadapter/docker-compose.windows-gpu.yml up -d --build
docker compose ps
docker compose logs -f freqtrade
```

### 4.3 One-off run command

```bash
docker compose -f docker-compose.yml -f ../quickadapter/docker-compose.windows-gpu.yml run --rm freqtrade trade \
  --config user_data/config.json \
  --strategy RLAgentStrategy \
  --freqaimodel ReforceXY
```

## 5. Hardware tuning guidance (7800X3D + RTX 4080 Super)

Good starting values:
- QuickAdapter:
  - `freqai.model_training_parameters.n_jobs = 8`
  - `freqai.data_kitchen_thread_count = 4`
- ReforceXY:
  - `freqai.model_training_parameters.device = "cuda"`
  - keep `rl_config.cpu_count = 4` and `rl_config.n_envs = 8` initially

If system is stable and cool, increase gradually. Do not change many knobs at once.

## 5.1 Current QuickAdapter dry-run profile (local)

Current working stack (GPU + dry-run + allpairs):

```bash
docker exec qa_dryrun_allpairs sh -lc 'tr "\0" " " </proc/1/cmdline; echo'
```

Expected key runtime settings:
- `dry_run = true`
- `freqai.model_training_parameters.device = "cuda"`
- `freqai.model_training_parameters.n_jobs = 8`
- `freqai.data_kitchen_thread_count = 4`
- Strategy hard stoploss floor in `QuickAdapterV3`: `stoploss = -0.99`
- Custom stoploss remains enabled (`use_custom_stoploss = true`)

## 6. Troubleshooting

### GPU not detected in container

1. Restart Docker Desktop.
2. Re-run GPU validation command from section 2.
3. Check you are using the GPU compose override (`-f ...windows-gpu.yml`).
4. For QuickAdapter, temporarily set `device` to `"cpu"` in `config.windows-gpu.json` to confirm non-GPU path works.

### CUDA out-of-memory / training crashes

Reduce concurrency:
- QuickAdapter: lower `n_jobs` from `8` to `4` and keep `data_kitchen_thread_count` at `4` or lower.
- ReforceXY: reduce `rl_config.n_envs` and/or `rl_config.cpu_count`.

### Clean stop

From each strategy directory:

```bash
docker compose down
```

 How to run Optuna properly (QuickAdapter only)

  1. Keep ReforceXY running as-is.
  2. Stop QuickAdapter dual-run profile:

  cd /home/vasko/freqai-foundry-strategies/quickadapter
  docker compose -f docker-compose.yml -f docker-compose.windows-gpu.yml -f docker-compose.wsl-top30-dualrun.yml down

  3. Start QuickAdapter Optuna profile:

  docker compose -f docker-compose.yml -f docker-compose.windows-gpu.yml -f docker-compose.wsl-top30-optuna.yml up -d --build

  4. Watch tuning progress:

  docker logs -f qa_dryrun_top30_optuna | rg -i "optuna|hyperopt|best params|starting training|done training"

  5. After session finishes (or after your timeout window), switch back to trading profile:

  docker compose -f docker-compose.yml -f docker-compose.windows-gpu.yml -f docker-compose.wsl-top30-optuna.yml down
  docker compose -f docker-compose.yml -f docker-compose.windows-gpu.yml -f docker-compose.wsl-top30-dualrun.yml up -d --build

   1. Watch these markers in logs:

  docker logs -f qa_dryrun_top30_optuna 2>&1 | rg -i "Starting training|Done training|Optuna .*objective hyperopt started|Optuna .*objective hyperopt completed"

  2. Completion signal is:

  - for each pair/namespace, you see Optuna ... started then Optuna ... completed
  - then Done training <PAIR>
  - then only normal heartbeat logs continue.