import pygame
import numpy as np
import cv2


def add_telemetry_overlay(rgb_array, obs):
    pygame.init()
    screen_width, screen_height = rgb_array.shape[1], rgb_array.shape[0]
    screen = pygame.Surface((screen_width, screen_height))
    font = pygame.font.Font(None, 24)

    telemetry = {
        "X Pos": obs[0],
        "Y Pos": obs[1],
        "X Vel": obs[2],
        "Y Vel": obs[3],
        "Angle": obs[4],
        "Angular Vel": obs[5],
        "Left Leg Contact": obs[6],
        "Right Leg Contact": obs[7],
    }

    frame = pygame.surfarray.make_surface(np.transpose(rgb_array, axes=(1, 0, 2)))
    screen.blit(frame, (0, 0))

    y_offset = 10
    for key, value in telemetry.items():
        text_surface = font.render(f"{key}: {value:.2f}", True, (255, 255, 255))
        screen.blit(text_surface, (10, y_offset))
        y_offset += 20

    # Convert Pygame Surface back to an RGB array
    telemetry_frame = pygame.surfarray.array3d(screen)
    return np.transpose(telemetry_frame, axes=(1, 0, 2))

def add_success_failure_to_frames(frames, success):
    overlay_text = "Landed" if success else "Crashed"
    color = (255, 255, 255) if success else (255, 255, 255)

    for i in range(len(frames)):
        # Convert frame to BGR for OpenCV processing
        bgr_frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)

        # Position for overlay text (top-right corner)
        text_size = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = bgr_frame.shape[1] - text_size[0] - 10
        text_y = 20  # Fixed y-offset for top-right corner

        # Add text overlay
        cv2.putText(bgr_frame, overlay_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Convert frame back to RGB
        frames[i] = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    return frames