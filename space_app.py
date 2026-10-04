"""
HuggingFace Space — واجهة مراقبة فقط.
لا تنفّذ صفقات ولا تحمل مفاتيح تداول.
يقرأ من storage/ المرفوعة (DuckDB + Parquet + JSON).
"""
from __future__ import annotations
import os
import gradio as gr
import pandas as pd
from data.storage import Storage

STORAGE_ROOT = os.environ.get("STORAGE_ROOT", "./storage")
storage = Storage(STORAGE_ROOT)


def get_status():
    try:
        trades = storage.con.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(pnl_net),0) pnl FROM trades"
        ).df()
        proposals = storage.list_proposals().head(20)
        alerts = storage.con.execute(
            "SELECT severity, name, message, ts FROM alerts ORDER BY ts DESC LIMIT 20"
        ).df()
        return {
            "trades": trades.to_dict(orient="records"),
            "proposals": proposals.to_dict(orient="records"),
            "alerts": alerts.to_dict(orient="records"),
        }
    except Exception as e:
        return {"error": str(e)}


def ui_dashboard():
    s = get_status()
    return pd.DataFrame(s.get("trades", [])), \
           pd.DataFrame(s.get("proposals", [])), \
           pd.DataFrame(s.get("alerts", []))


with gr.Blocks(title="Capital AI Brain — Monitor") as demo:
    gr.Markdown("# 🤖 Capital AI Brain\n**واجهة مراقبة فقط — لا تنفيذ**")
    btn = gr.Button("🔄 تحديث")
    with gr.Row():
        t = gr.Dataframe(label="Trades")
        p = gr.Dataframe(label="Proposals")
    a = gr.Dataframe(label="Alerts")
    btn.click(ui_dashboard, outputs=[t, p, a])
    demo.load(ui_dashboard, outputs=[t, p, a])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
