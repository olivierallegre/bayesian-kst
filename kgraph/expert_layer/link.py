class Link:

    def __init__(self, source, target):
        self.source = source
        self.target = target
        self.strength = 'strong'

    def set_strength(self, strength):
        self.strength = strength
