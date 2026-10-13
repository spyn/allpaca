#!/usr/bin/env bash
# Copy this working tree's worker sources to a machine, build allpaca-rpc-server
# there and (optionally) restart it. The worker then has this tree's build id,
# so the coordinator's "built from other sources" warning goes away.
#
#   scripts/deploy-worker.sh [options] <ssh-target> [remote-dir]
#
#   --features LIST   cargo features for the worker (default: hip)
#   --start ARGS      restart the worker with these arguments, e.g.
#                     --start "--device hip --insecure-bind -H 192.168.69.2 -p 50052"
#   --env "K=V ..."   environment for the build and the worker, e.g.
#                     --env "LD_LIBRARY_PATH=/opt/rocm/lib"
#   --log FILE        worker log on the remote machine (default: ~/allpaca-worker.log)
#
# ALLPACA_DEPLOY_SSH overrides the ssh command (identity file, options).
# ALLPACA_RPC_TOKEN, when set here, is passed to the restarted worker.
# The remote directory (default ~/allpaca-worker) holds only a copy: nothing
# is deleted there, and a git checkout elsewhere on the machine is not touched.
set -euo pipefail

features=hip start="" env_vars="" log='~/allpaca-worker.log'
while [ $# -gt 0 ]; do
    case "$1" in
        --features) features=$2; shift 2 ;;
        --start) start=$2; shift 2 ;;
        --env) env_vars=$2; shift 2 ;;
        --log) log=$2; shift 2 ;;
        -h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "unknown option $1" >&2; exit 2 ;;
        *) break ;;
    esac
done
[ $# -ge 1 ] || { echo "usage: $0 [options] <ssh-target> [remote-dir]" >&2; exit 2; }
target=$1
dir=${2:-'~/allpaca-worker'}
ssh_cmd=${ALLPACA_DEPLOY_SSH:-ssh}

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
# Tracked files plus new ones that are not ignored: a source file added since
# the last commit must go too.
files=$(git ls-files -co --exclude-standard -- Cargo.toml Cargo.lock rust-toolchain.toml crates hip cuda tools scripts)
echo "copying $(echo "$files" | wc -l) files to $target:$dir"
echo "$files" | tar -cf - -T - | $ssh_cmd "$target" "mkdir -p $dir && cd $dir && tar -xf -"

echo "building allpaca-rpc-server --features $features"
$ssh_cmd "$target" "cd $dir && export PATH=\$HOME/.cargo/bin:\$PATH $env_vars && \
    cargo build --release -p allpaca-rpc-server --features $features > deploy-build.log 2>&1; rc=\$?; \
    grep -E 'warning: allpaca-[a-z]+@|Finished' deploy-build.log; \
    if [ \$rc -ne 0 ]; then grep -E -A12 '^error' deploy-build.log | head -60; exit 1; fi; \
    echo \"worker build id: \$(./target/release/allpaca-rpc-server --build-id)\""

if [ -n "$start" ]; then
    echo "restarting the worker"
    # stdin, stdout and stderr all leave the ssh session, or ssh waits on the worker.
    $ssh_cmd "$target" "pid=\$(pidof allpaca-rpc-server) && kill \$pid && sleep 2; cd $dir && \
        export ALLPACA_RPC_TOKEN='${ALLPACA_RPC_TOKEN:-}' $env_vars && \
        setsid -f ./target/release/allpaca-rpc-server $start > $log 2>&1 < /dev/null; \
        sleep 2; pidof allpaca-rpc-server > /dev/null && tail -n 2 $log"
fi
