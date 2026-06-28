from .base_agent import BaseAgent


class DalioAgent(BaseAgent):
    def __init__(self):
        super().__init__("Ray Dalio", "ray_dalio.md")
