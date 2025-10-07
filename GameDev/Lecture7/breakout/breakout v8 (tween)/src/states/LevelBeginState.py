from src.states.BaseState import BaseState
from src.constants import *
from src.resources import *
from src.Dependency import *
import src.tween.tween as tween
import src.CommonRender as CommonRender

class LevelBeginState(BaseState):
    def __init__(self):
        super(LevelBeginState, self).__init__()

        self.transition_alpha = 0

        self.level_label_y = -96

        self.nothing=0

    def Exit(self):
        pass

    def Enter(self, params):
        self.paddle = params["paddle"]
        self.bricks = params["bricks"]
        self.health = params["health"]
        self.score = params["score"]
        self.high_scores = params["high_scores"]
        self.level = params["level"]
        self.recover_points = params["recover_points"]
        self.ball = params["ball"]

        def game_start():
            g_state_manager.Change('play', {
                'paddle':self.paddle,
                'level':self.level,
                'health':self.health,
                'score':self.score,
                'high_scores': self.high_scores,
                'ball':self.ball,
                'recover_points': self.recover_points,
                'bricks':self.bricks
            })

        def pause_1sec():
            tween.to(self, 'nothing', 0, 1, ease_type='linear').on_complete(disappear_to_bottom)

        def disappear_to_bottom():
            tween.to(self, 'transition_alpha', 0, 0.25, ease_type='linear')
            tween.to(self, 'level_label_y', HEIGHT + 90, 0.25, ease_type='linear').on_complete(game_start)

        tween.to(self, 'transition_alpha', 255, 1, ease_type='linear')
        tween.to(self, 'level_label_y', HEIGHT / 2 - 24, 0.25, ease_type='linear').on_complete(pause_1sec)

    def update(self, dt, events):
        pass

    def render(self, screen):
        self.paddle.render(screen)
        self.ball.render(screen)

        for brick in self.bricks:
            brick.render(screen)

        CommonRender.RenderScore(screen, self.score)
        CommonRender.RenderHealth(screen, self.health)

        s = pygame.Surface((WIDTH, 144))
        s.set_alpha(self.transition_alpha)
        s.fill((95, 205, 228))
        screen.blit(s, (0, self.level_label_y-60))

        t_level = gFonts['large'].render("Level: " + str(self.level), False, (255, 255, 255))
        rect = t_level.get_rect(center=(WIDTH/2, self.level_label_y))
        screen.blit(t_level, rect)



        #pygame.draw.rect(screen, (95, 205, 228, 200), pygame.Rect(0, self.level_label_y - 24, WIDTH, 144))
