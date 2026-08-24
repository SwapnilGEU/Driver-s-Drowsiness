from twilio.rest import Client
from config.settings import (
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_PHONE_NUMBER
)

client = Client(
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN
)


def send_sms(phone_number, sms_message):

    try:

        client.messages.create(
            body=sms_message,
            from_=TWILIO_PHONE_NUMBER,
            to=phone_number
        )

        print("SMS Sent")

    except Exception as e:
        print("SMS Error:", e)



def make_call(phone_number, voice_message):

    try:

        twiml = f"""
        <Response>
            <Say voice='alice'>
                {voice_message}
            </Say>
        </Response>
        """

        client.calls.create(
            twiml=twiml,
            to=phone_number,
            from_=TWILIO_PHONE_NUMBER
        )

        print("Call Sent")

    except Exception as e:
        print("Call Error:", e)



def send_emergency_alert(phone, voice, sms):

    send_sms(phone, sms)
    make_call(phone, voice)