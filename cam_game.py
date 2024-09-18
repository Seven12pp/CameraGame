# Made by Seven12

import cv2
import mediapipe as mp
import pygame
import random
import numpy as np
import time
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2)

pygame.init()
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

ball_radius = 20
ball_speed = 5
balls = []
purple_ball_chance = 0.1
purple_ball_radius = 25
laser_color = (255, 0, 0)
laser_active = False
last_shot_time = 0
cooldown = 10
laser_speed = 10
laser_width = 10
lasers = []

def create_ball():
    x = random.randint(0, screen_width - ball_radius)
    y = 0
    color = (255, 0, 0)
    points = 1
    radius = ball_radius
    if random.random() < purple_ball_chance:
        color = (128, 0, 128)
        points = 2
        radius = purple_ball_radius
    return [x, y, color, points, radius]

def check_collision(finger_x, finger_y, ball):
    ball_x, ball_y, _, _, ball_radius = ball
    dist = ((finger_x - ball_x) ** 2 + (finger_y - ball_y) ** 2) ** 0.5
    return dist < ball_radius

cap = cv2.VideoCapture(0)
score = 0

def is_hand_pistol(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    thumb_folded = landmarks[3].y > thumb_tip.y
    index_folded = landmarks[7].y > index_tip.y
    middle_folded = landmarks[11].y < middle_tip.y
    ring_folded = landmarks[15].y < ring_tip.y
    pinky_folded = landmarks[19].y < pinky_tip.y

    return thumb_folded and index_folded and middle_folded and ring_folded and pinky_folded

def shoot_laser(index_pos, direction):
    lasers.append({"pos": index_pos, "direction": direction})

def move_lasers():
    global lasers, balls, score
    for laser in lasers[:]:
        laser["pos"][0] += laser["direction"][0] * laser_speed
        laser["pos"][1] += laser["direction"][1] * laser_speed

        pygame.draw.line(screen, laser_color, laser["pos"], 
                         (laser["pos"][0] + laser["direction"][0] * 20, laser["pos"][1] + laser["direction"][1] * 20), laser_width)

        if laser["pos"][0] < 0 or laser["pos"][0] > screen_width or laser["pos"][1] < 0 or laser["pos"][1] > screen_height:
            lasers.remove(laser)

        for ball in balls[:]:
            if check_collision(laser["pos"][0], laser["pos"][1], ball):
                balls.remove(ball)
                score += ball[3]

def draw_reload_bar(current_time, last_shot_time, cooldown):
    bar_width = 200
    bar_height = 20
    bar_x = 20
    bar_y = screen_height - 40

    time_since_shot = current_time - last_shot_time
    if time_since_shot >= cooldown:
        time_since_shot = cooldown

    fill_width = int((time_since_shot / cooldown) * bar_width)

    pygame.draw.rect(screen, (255, 255, 255), (bar_x, bar_y, bar_width, bar_height), 2)
    pygame.draw.rect(screen, (0, 255, 0), (bar_x, bar_y, fill_width, bar_height))

running = True
while running:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (screen_width, screen_height))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if random.randint(1, 20) == 1:
        balls.append(create_ball())

    balls = [[x, y + ball_speed, color, points, radius] for x, y, color, points, radius in balls]
    balls = [ball for ball in balls if ball[1] < screen_height]

    screen.blit(frame_surface, (0, 0))

    for ball in balls:
        ball_x, ball_y, color, _, radius = ball
        pygame.draw.circle(screen, color, (ball_x, ball_y), radius)

    current_time = time.time()

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            finger_points = [
                hand_landmarks.landmark[4],
                hand_landmarks.landmark[8],
                hand_landmarks.landmark[12],
                hand_landmarks.landmark[16],
                hand_landmarks.landmark[20],
            ]

            for finger in finger_points:
                finger_x = int((1 - finger.x) * screen_width)
                finger_y = int(finger.y * screen_height)
                pygame.draw.circle(screen, (0, 255, 0), (finger_x, finger_y), 10)

                for ball in balls[:]:
                    if check_collision(finger_x, finger_y, ball):
                        balls.remove(ball)
                        score += ball[3]

            if is_hand_pistol(hand_landmarks.landmark):
                if current_time - last_shot_time > cooldown:
                    index_tip = hand_landmarks.landmark[8]
                    index_base = hand_landmarks.landmark[5]

                    index_tip_screen = [int((1 - index_tip.x) * screen_width), int(index_tip.y * screen_height)]
                    index_base_screen = [int((1 - index_base.x) * screen_width), int(index_base.y * screen_height)]

                    dx = index_tip_screen[0] - index_base_screen[0]
                    dy = index_tip_screen[1] - index_base_screen[1]
                    magnitude = math.sqrt(dx ** 2 + dy ** 2)

                    if magnitude != 0:
                        direction = [dx / magnitude, dy / magnitude]
                    else:
                        direction = [0, -1]

                    shoot_laser(index_tip_screen, direction)
                    last_shot_time = current_time

    move_lasers()

    font = pygame.font.Font(None, 36)
    text = font.render(f'Score: {score}', True, (0, 0, 0))
    screen.blit(text, (screen_width // 2 - 50, 10))

    draw_reload_bar(current_time, last_shot_time, cooldown)

    pygame.display.flip()
    clock.tick(30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

cap.release()
pygame.quit()
