import winsound


def play_alarm():

    try:
        winsound.PlaySound(
            "WakeSafeAI\assets\alarm.wav",
            winsound.SND_FILENAME | winsound.SND_ASYNC
        )

    except Exception as e:
        print("Alarm Error:", e)