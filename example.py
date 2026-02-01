from flwr.server.strategy import FedAvg

from fedbench.client.registry import ClientRegistry
from fedbench.common import MLRuntime
from fedbench.server.registry import FlwrStrategyRegistry


client = ClientRegistry(MLRuntime.NUMPY)

@client.synthesizer
def synthesizer():
    return None


server = FlwrStrategyRegistry(MLRuntime.NUMPY)

@server.flwr_strategy
def fed_avg() -> FedAvg:
    return FedAvg()
