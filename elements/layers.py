import gum

class NetworkLayer:

    def __init__(self, network):
        self.network = network
        self.nodes = network.nodes


class PriorNetworkLayer(NetworkLayer):

    def __init__(self):
        self._set_layer_internal_links(self.network)

    def _set_layer_internal_links(self, network):
