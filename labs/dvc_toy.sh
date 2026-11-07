#!/usr/bin/env bash
# Task 1.4 toy part - DVC basics on a tiny CSV, in a NEW empty folder.
# Run it step by step (copy each block into the terminal) and screenshot each step.
#
# Usage:  bash labs/dvc_toy.sh [dagshub-user]
#   with a DagsHub user: pushes Git to a DagsHub repo "dvc-toy-24L-0672" and data to its DVC storage
#   without:             uses a local folder as the DVC remote
set -euo pipefail
DVC="${DVC:-$(cd "$(dirname "$0")/.." && pwd)/.venv/bin/dvc}"
USER_DH="${1:-}"
TOY=~/dvc-toy-24L-0672
rm -rf "$TOY" && mkdir -p "$TOY" && cd "$TOY"
echo "### 1. init Git + DVC in $TOY"
git init -q -b main && "$DVC" init -q && git commit -qm "init dvc" && git log --oneline

echo "### 2. create a small CSV and track it with DVC (not Git)"
printf "id,label\n1,real\n2,ai\n3,real\n" > data.csv
"$DVC" add data.csv
cat data.csv.dvc          # the pointer file: md5 hash, size, path
cat .gitignore            # DVC told Git to ignore the real data
git add data.csv.dvc .gitignore && git commit -qm "data v1 (3 rows)"

echo "### 3. change the data -> version 2"
printf "4,ai\n5,ai\n" >> data.csv
"$DVC" add data.csv
git add data.csv.dvc && git commit -qm "data v2 (5 rows)"
git log --oneline

echo "### 4. go back to the old version"
git checkout -q HEAD~1 -- data.csv.dvc && "$DVC" checkout data.csv
echo "rows now: $(($(wc -l < data.csv) - 1))  (v1)"; cat data.csv
git checkout -q main -- data.csv.dvc && "$DVC" checkout data.csv
echo "rows now: $(($(wc -l < data.csv) - 1))  (back to v2)"

echo "### 5. configure a remote, push, delete local data, pull it back"
if [[ -n "$USER_DH" ]]; then
  # Create an EMPTY repo named dvc-toy-24L-0672 on dagshub.com first.
  git remote add origin "https://dagshub.com/$USER_DH/dvc-toy-24L-0672.git"
  "$DVC" remote add -d origin "https://dagshub.com/$USER_DH/dvc-toy-24L-0672.dvc"
  "$DVC" remote modify origin --local auth basic
  "$DVC" remote modify origin --local user "$USER_DH"
  read -rsp "DagsHub token (input hidden, stored only in .dvc/config.local): " TOKEN; echo
  "$DVC" remote modify origin --local password "$TOKEN"
  git add .dvc/config && git commit -qm "dvc remote: dagshub"
  git push -u origin main   # asks for DagsHub username + token
else
  mkdir -p /tmp/dvc-toy-remote
  "$DVC" remote add -d localremote /tmp/dvc-toy-remote
  git add .dvc/config && git commit -qm "dvc remote: local folder"
fi
"$DVC" push
rm -f data.csv && rm -rf .dvc/cache
echo "deleted: $(ls data.csv 2>&1 | tail -1)"
"$DVC" pull
echo "pulled back:"; cat data.csv
