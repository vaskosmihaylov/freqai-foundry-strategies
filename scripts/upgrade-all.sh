#!/usr/bin/env sh
  set -eu

  cd /home/vasko.mihaylov/freqai-foundry-strategies/quickadapter
  FREQTRADE_CONFIG=./user_data/config.json \
  LOCAL_DOCKER_IMAGE=quickadapter-freqtrade \
  REMOTE_DOCKER_IMAGE=freqtradeorg/freqtrade:stable_freqai \
  ./docker-upgrade.sh >> user_data/logs/docker-upgrade.log 2>&1

  cd /home/vasko.mihaylov/freqai-foundry-strategies/ReforceXY
  FREQTRADE_CONFIG=./user_data/config.json \
  LOCAL_DOCKER_IMAGE=reforcexy-freqtrade \
  REMOTE_DOCKER_IMAGE=freqtradeorg/freqtrade:stable_freqairl \
  ./docker-upgrade.sh >> user_data/logs/docker-upgrade.log 2>&1