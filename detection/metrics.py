import time

frame_count = 0
start_time = time.time()
fps = 0


def calculate_fps():
    global frame_count, start_time, fps

    frame_count += 1

    elapsed = time.time() - start_time

    if elapsed >= 1:
        fps = frame_count / elapsed
        frame_count = 0
        start_time = time.time()

    return fps


def calculate_distance(pixel_width, known_width, focal_length):

    if pixel_width <= 0:
        return 100

    distance = (known_width * focal_length) / pixel_width

    return distance