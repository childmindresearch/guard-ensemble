#!/usr/bin/env bash

set -eou pipefail

SRC=$(docker volume inspect hf-cache --format '{{ .Mountpoint }}')

echo "Bootstrap will:"
echo "- sync: $SRC -> /opt/dlami/nvme/hf-cache"
echo "- wipe: /tmp/svim-k -> /tmp/svim-k (new)"
echo
echo "Enter 'yes' to continue"

read -rp "> " reply

if [ "$reply" != "yes" ]; then
  exit
fi

sudo rm -rf /tmp/svim-k && mkdir /tmp/svim-k

echo "> Wiped /tmp/svim-k"

sudo rsync -aH --info=progress2 "$SRC"/ /opt/dlami/nvme/hf-cache/
sudo chown -R "$(id -u):$(id -g)" /opt/dlami/nvme/hf-cache

echo "> Synced /opt/dlami/nvme/hf-cache and assigned ownership"
