import cv2


def draw_status(frame, is_drowsy):

    color = (0, 0, 255) if is_drowsy else (0, 255, 0)

    cv2.circle(frame, (25, 50), 10, color, -1)

    cv2.putText(
        frame,
        "DROWSY" if is_drowsy else "NORMAL",
        (45, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )


def draw_fps(frame, fps):

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


def draw_distance(frame, distance):

    cv2.putText(
        frame,
        f"Distance: {distance:.1f} cm",
        (10, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2
    )


def draw_awake_bar(frame, awake_percentage):

    bar_x = 10
    bar_width = 350
    bar_height = 25

    bar_y = frame.shape[0] - 40

    filled_width = int((awake_percentage / 100) * bar_width)

    if awake_percentage > 60:
        bar_color = (0, 255, 0)
    elif awake_percentage > 30:
        bar_color = (0, 255, 255)
    else:
        bar_color = (0, 0, 255)

    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (50, 50, 50),
        -1
    )

    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + filled_width, bar_y + bar_height),
        bar_color,
        -1
    )

    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Awake Level: {awake_percentage}%",
        (10, bar_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )