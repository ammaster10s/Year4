from src.states.BaseState import BaseState

class StateMachine:
    def __init__(self, screen):
        self.screen = screen
        self.current = None

    def SetStates(self, states):
        self.states = states

    def Change(self, state_name, enter_params):
        assert(self.states[state_name]) #state is exist!! right?
        if self.current:
            self.current.Exit()
        self.current = self.states[state_name]
        self.current.Enter(enter_params)

    def update(self, dt, events):
        self.current.update(dt, events)

    def render(self):
        self.current.render(self.screen)

