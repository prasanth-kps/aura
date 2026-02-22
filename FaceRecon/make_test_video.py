import cv2
import numpy as np

w, h, fps, secs = 640, 360, 20, 6
out = cv2.VideoWriter("test_input.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

for i in range(fps * secs):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        f"Test Video Frame {i}",
        (60, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    out.write(frame)

out.release()
print("Generated test_input.mp4")
