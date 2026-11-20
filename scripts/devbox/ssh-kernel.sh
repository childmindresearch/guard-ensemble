#!/usr/bin/env bash

# ssh-kernel.sh - launch devbox Jupyter kernel on an EC2 instance.
#
# Usage:
#
#   ssh-kernel.sh <ssh-public-key|age-config> <env-file> <host>
#
# Requirements:
#
# - SSH
# - AWS CLI
# - age (https://github.com/FiloSottile/age)
# - encrypted `.env.kernel.age` file
# - guard-ensemble Docker image built on host
#
# ---
#
# Configure SSH with AWS SSM (~/.ssh/config)
#
#   Host devbox
#     HostName i-123456789abcdefgh # EC2 instance ID
#     User ubuntu
#     IdentityFile  ~/.ssh/id_ed25519 # Your private key
#     IdentitiesOnly yes
#     ProxyCommand aws ssm start-session --target %h --document-name AWS-StartSSHSession --parameters portNumber=%p
#
# ---
#
# Create .env.kernel outside of repo (~/.private/.env.kernel)
#
#   MIRROR_DATA_ACCESS_KEY_ID=xxxxxx
#   MIRROR_DATA_SECRET_ACCESS_KEY=xxxxxx
#   MIRROR_DATA_BUCKET=xxxxxx
#
# ---
#
# Encrypt .env.kernel with `age`
#
#   $ cd ~/.private
#   $ age -R ~/.ssh/id_ed25519.pub .env.kernel > .env.kernel.age
#   $ rm .env.kernel
#
# ---
#
# Run ssh-kernel.sh
#
#   $ ./scripts/devbox/ssh-kernel.sh ~/.ssh/id_ed25519.pub ~/.private/.mirror.env.age devbox /home/ubuntu/sync/guard-ensemble
#

set -eou pipefail

KEY="$1"
ENVFILE="${2:-$HOME/.private/.env.kernel.age}"
TO_HOST="${3:-devbox}"
REPO_DIR="${4:-/home/ubuntu/sync/guard-ensemble}"

env_contents=$(age -d -i "$KEY" "$ENVFILE")
remote_cmd="set -a; . /dev/stdin; set +a; cd -- $(printf '%q' "$REPO_DIR") && exec ./scripts/devbox/run-kernel.sh"

printf '%s\n' "$env_contents" | ssh -- "$TO_HOST" "$remote_cmd"
