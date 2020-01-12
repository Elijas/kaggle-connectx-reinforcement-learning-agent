#!/bin/bash -e

## Parse arguments
print_usage() {
  printf "
Uploads a given agent to kaggle competition.

Usage: ./%s -a AGENT_NAME
Options:
    -a      Agent name. Giving 'agent_name' will upload 'agents/agent_name.py'
" "$(basename "$0")"
}

AGENT_NAME=''

while getopts 'a:' flag; do
  case "${flag}" in
  a) AGENT_NAME="$OPTARG" ;;
  *)
    print_usage
    exit 1
    ;;
  esac
done

if [ -z "$AGENT_NAME" ]; then
  print_usage
  exit 1
fi

## Set Current Dir to the script's dir
cd "${0%/*}"

## Make sure that the agent doesn't crash in a game
python -m tools.check_sanity "$AGENT_NAME"

## Deploy the agent
MSG=$(date +%Y-%m-%d_%H-%M-%S)
kaggle competitions submit -c connectx -f ./agents/"$AGENT_NAME".py -m "$MSG"
printf "\n[%s] Successfully deployed: %s" "$AGENT_NAME" "$MSG"