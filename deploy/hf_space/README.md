---
title: 24L-0672_A2_Ops VisionGuard API
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# VisionGuard model API (24L-0672_A2_Ops)

AI-generated vs real image classifier (MobileNetV2 fine-tuned, TensorFlow/Keras).

- `GET /health` model version, threshold, uptime
- `POST /predict` multipart field `image`, header `X-API-Key`

```bash
curl -H "X-API-Key: $MODEL_API_KEY" -F image=@photo.jpg https://<user>-24l-0672-a2-ops.hf.space/predict
```
