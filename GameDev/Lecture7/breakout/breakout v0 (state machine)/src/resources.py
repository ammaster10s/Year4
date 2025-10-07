import pygame
from src.StateMachine import StateMachine

g_state_manager = StateMachine()

gFonts = {
        'small': pygame.font.Font('./fonts/font.ttf', 24),
        'medium': pygame.font.Font('./fonts/font.ttf', 48),
        'large': pygame.font.Font('./fonts/font.ttf', 96)
}

gSounds = {
    'paddle-hit': pygame.mixer.Sound('sounds/paddle_hit.wav'),
}

