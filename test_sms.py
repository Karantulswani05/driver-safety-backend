from sms_service import send_sms

success, response = send_sms(
    "+919529775314",     # Your verified number
    "🚨 Driver Safety System Test SMS\n\nIf you received this message, Twilio integration is working."
)

print(success)
print(response)