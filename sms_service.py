import os
from pathlib import Path
from dotenv import load_dotenv
import requests

load_dotenv(Path(__file__).with_name(".env"))

FAST2SMS_API_KEY = os.getenv("FAST2SMS_API_KEY")


def send_sms(phone_number, message):

    url = "https://www.fast2sms.com/dev/bulkV2"

    headers = {
        "Authorization": FAST2SMS_API_KEY
    }

    data = {
        "route": "q",
        "message": message,
        "numbers": phone_number
    }

    try:

        response = requests.post(
            url,
            headers=headers,
            data=data,
            timeout=15
        )

        print(response.json())

        if response.status_code == 200:

            return True, response.json()

        return False, response.text

    except Exception as e:

        return False, str(e)