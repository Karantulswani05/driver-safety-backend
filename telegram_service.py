import requests

# ==========================
# TELEGRAM CONFIG
# ==========================

BOT_TOKEN = "8807955031:AAG6UIKVweBTbnTjwe1NxMbvs4NOvQMBxqQ"
BOT_USERNAME = "ai_driver_safety_bot"
TELEGRAM_API_BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"


def get_telegram_start_link(driver_id, contact_id):
    if not BOT_USERNAME:
        return False, "Telegram BOT_USERNAME is missing."

    if not contact_id:
        return False, "Telegram contact ID is missing."

    payload = f"{driver_id}__{contact_id}"
    return True, f"https://t.me/{BOT_USERNAME}?start={payload}"


def get_telegram_start_connections(driver_id):
    if not BOT_TOKEN:
        return False, "Telegram BOT_TOKEN is missing."

    try:
        requests.post(f"{TELEGRAM_API_BASE_URL}/deleteWebhook", timeout=10)
        response = requests.get(
            f"{TELEGRAM_API_BASE_URL}/getUpdates",
            params={"allowed_updates": '["message"]'},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        return False, f"Telegram updates failed: {error}"

    data = response.json()

    if not data.get("ok"):
        return False, data.get("description", "Telegram updates failed.")

    connections_by_contact = {}

    for update in data.get("result", []):
        message = update.get("message") or {}
        text = message.get("text", "")
        chat = message.get("chat") or {}

        if not text.startswith("/start "):
            continue

        payload = text.split(maxsplit=1)[1]

        if "__" not in payload:
            continue

        update_driver_id, contact_id = payload.split("__", 1)

        if update_driver_id != driver_id:
            continue

        chat_id = chat.get("id")

        if not chat_id:
            continue

        telegram_name = " ".join(
            str(part)
            for part in [chat.get("first_name"), chat.get("last_name")]
            if part
        ) or str(chat.get("username") or "")

        connections_by_contact[contact_id] = {
            "contactId": contact_id,
            "telegramChatId": str(chat_id),
            "telegramName": telegram_name,
        }

    return True, list(connections_by_contact.values())


def send_telegram_alert(chat_id, message):

    print("=" * 50)
    print("Sending Telegram")
    print("Destination Chat ID:", chat_id)
    print("=" * 50)

    url = f"{TELEGRAM_API_BASE_URL}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": message
    }

    response = requests.post(url, json=payload)

    print(response.text)

    return response.json()
