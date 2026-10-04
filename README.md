---
title: Capital AI Brain Monitor
emoji: 🤖
colorFrom: blue
colorTo: gray
sdk: gradio
app_file: space_app.py
pinned: false
---

# Capital AI Brain — Monitor Only

⚠️ **تحذير**: هذه الواجهة للعرض فقط. لا تتصل بـ Capital.com،
لا تنفّذ صفقات، ولا تحمل مفاتيح حقيقية.

## التشغيل الكامل (محلي / VPS)
```bash
git clone <private-repo>
cd capital_ai_brain
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env    # املأ المفاتيح الحقيقية
python main.py
