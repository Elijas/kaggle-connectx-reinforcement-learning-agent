#!/bin/bash -e

## Parse arguments
print_usage() {
  printf "
Uploads a given agent to kaggle competition.

Usage: ./%s -a AGENT_NAME
Options:
    -a      Agent name
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
./play.py "$AGENT_NAME" "$AGENT_NAME"

## Deploy the agent
MSG=$(date +%Y-%m-%d_%H-%M-%S)
kaggle competitions submit -c connectx -f ./agents/"$AGENT_NAME".py -m "$MSG"
printf "\n[%s] Successfully deployed: %s\n" "$AGENT_NAME" "$MSG"
