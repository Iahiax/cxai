from __future__ import annotations
import json, time, html
from dataclasses import dataclass, field
from typing import Callable, Optional
import requests
from data.storage import Storage


@dataclass
class TelegramConfig:
    bot_token: str
    channel_id: str
    allowed_user_ids: list[int] = field(default_factory=list)
    admin_user_ids: list[int] = field(default_factory=list)
    rate_limit_per_min: int = 30
    dangerous_commands: tuple[str, ...] = ("/live", "/close_all", "/kill", "/leverage")
    confirm_ttl_seconds: int = 60


class TelegramBot:
    def __init__(self, cfg: TelegramConfig, storage: Storage):
        self.cfg = cfg
        self.storage = storage
        self.base = f"https://api.telegram.org/bot{cfg.bot_token}"
        self._last_call: list[float] = []
        self._pending_confirm: dict[int, dict] = {}
        self._handlers: dict[str, Callable] = {}
        self._offset = 0

    def _rate_limit(self):
        now = time.time()
        self._last_call = [t for t in self._last_call if now - t < 60]
        if len(self._last_call) >= self.cfg.rate_limit_per_min:
            time.sleep(1.5)
        self._last_call.append(now)

    def send(self, text: str, parse_mode: str = "HTML"):
        self._rate_limit()
        r = requests.post(f"{self.base}/sendMessage", json={
            "chat_id": self.cfg.channel_id,
            "text": text, "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }, timeout=15)
        r.raise_for_status()
        return r.json()

    def _is_allowed(self, user_id: int) -> bool:
        return (not self.cfg.allowed_user_ids) or (user_id in self.cfg.allowed_user_ids)

    def _is_admin(self, user_id: int) -> bool:
        return user_id in self.cfg.admin_user_ids

    def command(self, name: str):
        def deco(fn):
            self._handlers[name] = fn
            return fn
        return deco

    def _audit(self, actor: str, action: str, payload: dict, result: str):
        self.storage.con.execute("""
            INSERT INTO audit_log (id, actor, action, payload, result) VALUES
            ((SELECT COALESCE(MAX(id),0)+1 FROM audit_log), ?, ?, ?, ?)
        """, [actor, action, json.dumps(payload, default=str), result])

    def run_forever(self, stop_flag: Callable[[], bool] = lambda: False):
        while not stop_flag():
            try:
                r = requests.get(f"{self.base}/getUpdates",
                                 params={"offset": self._offset + 1, "timeout": 25},
                                 timeout=35)
                for u in r.json().get("result", []):
                    self._offset = max(self._offset, u["update_id"])
                    self._dispatch(u)
            except Exception as e:
                print(f"[telegram] poll error: {e}")
                time.sleep(3)

    def _dispatch(self, update: dict):
        msg = update.get("message") or update.get("edited_message")
        if not msg:
            return
        user_id = msg["from"]["id"]
        text = (msg.get("text") or "").strip()
        if not text.startswith("/"):
            return
        if not self._is_allowed(user_id):
            self._audit(str(user_id), text, {}, "denied_not_allowed")
            return
        cmd = text.split()[0].split("@")[0].lower()

        if cmd in self.cfg.dangerous_commands:
            if not self._is_admin(user_id):
                self._audit(str(user_id), cmd, {}, "denied_not_admin")
                return
            pending = self._pending_confirm.get(user_id)
            if not pending or pending["cmd"] != cmd or \
               time.time() - pending["ts"] > self.cfg.confirm_ttl_seconds:
                code = f"{int(time.time()) % 100000:05d}"
                self._pending_confirm[user_id] = {
                    "cmd": cmd, "code": code, "ts": time.time()
                }
                self.send(f"⚠️ أمر حساس: <code>{html.escape(cmd)}</code>\n"
                          f"للتأكيد أرسل: <code>{cmd} confirm {code}</code>")
                return
            if f"confirm {pending['code']}" not in text:
                self.send("❌ رمز التأكيد غير صحيح.")
                return
            self._pending_confirm.pop(user_id, None)

        handler = self._handlers.get(cmd)
        if not handler:
            self.send(f"❓ أمر غير معروف: <code>{html.escape(cmd)}</code>")
            return
        try:
            result = handler(user_id, text) or "✅ تم"
            self._audit(str(user_id), cmd, {"text": text}, "ok")
            self.send(result)
        except Exception as e:
            self._audit(str(user_id), cmd, {"text": text}, f"error:{e}")
            self.send(f"💥 خطأ: <code>{html.escape(str(e))}</code>")


def register_default_commands(bot: TelegramBot, storage: Storage, risk_engine=None):
    @bot.command("/start")
    def _start(uid, txt):
        return ("🤖 <b>Capital AI Brain</b> جاهز.\n"
                "/حالة — /صفقات — /مخاطر — /اقتراحات — /ايقاف")

    @bot.command("/حالة")
    def _status(uid, txt):
        n = storage.con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        return f"📊 الصفقات: <b>{n}</b>"

    @bot.command("/صفقات")
    def _trades(uid, txt):
        df = storage.con.execute(
            "SELECT trade_id, side, pnl_net FROM trades ORDER BY entry_ts DESC LIMIT 10"
        ).df()
        if df.empty:
            return "لا توجد صفقات."
        return "📈 آخر الصفقات:\n" + "\n".join(
            f"• {r.side} — {r.pnl_net:+.2f}" for r in df.itertuples()
        )

    @bot.command("/اقتراحات")
    def _proposals(uid, txt):
        df = storage.list_proposals("pending")
        if df.empty:
            return "لا اقتراحات معلّقة."
        return "\n".join(f"#{r.id} [{r.kind}] {r.title}" for r in df.itertuples())

    if risk_engine:
        @bot.command("/kill")
        def _kill(uid, txt):
            risk_engine.trigger_kill("telegram_manual")
            return "🛑 Kill switch مُفعّل."

        @bot.command("/استئناف")
        def _release(uid, txt):
            risk_engine.release_kill()
            return "✅ تم تحرير Kill switch."
