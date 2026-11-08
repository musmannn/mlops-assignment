#!/usr/bin/env bash
# Deploy the VisionGuard model API to Vercel (Task 7, pathway B).
#
# Prerequisites: `npx vercel login` once; models/champion.onnx (python -m training.export_onnx);
# MODEL_API_KEY in .env.
#
#   bash deploy/vercel/deploy.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$ROOT/deploy/vercel"
BUILD="$HERE/.build"            # git-ignored; keeps .vercel/project.json so redeploys hit the same project
PROJECT="24l-0672-a2-ops"       # Vercel project names must be lowercase
VERCEL="npx --yes vercel@latest"

# ---- 1. assemble a minimal build folder: API code + ONNX model, no TensorFlow ----
mkdir -p "$BUILD"
find "$BUILD" -mindepth 1 -maxdepth 1 ! -name .vercel -exec rm -rf {} +
cp -r "$HERE/api" "$HERE/vercel.json" "$HERE/requirements.txt" "$BUILD/"
echo "3.12" > "$BUILD/.python-version"
rsync -a --exclude "__pycache__" --exclude "web" "$ROOT/visionguard" "$BUILD/"
mkdir -p "$BUILD/models"
cp "$ROOT/models/champion.onnx" "$ROOT/models/champion.json" "$BUILD/models/"
echo "build folder: $(du -sh "$BUILD" | cut -f1)"

cd "$BUILD"
# ---- 2. link (creates the project on first run) ----
$VERCEL link --yes --project "$PROJECT" >/dev/null

# ---- 3. API key as an encrypted Vercel env var (read from .env, never echoed) ----
KEY="$(grep '^MODEL_API_KEY=' "$ROOT/.env" | cut -d= -f2-)"
[[ -n "$KEY" ]] || { echo "MODEL_API_KEY missing in .env"; exit 1; }
$VERCEL env rm MODEL_API_KEY production --yes >/dev/null 2>&1 || true
printf '%s' "$KEY" | $VERCEL env add MODEL_API_KEY production >/dev/null
echo "MODEL_API_KEY set on Vercel (production)"

# ---- 4. production deployment ----
$VERCEL deploy --prod --yes
