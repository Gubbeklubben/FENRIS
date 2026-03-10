from fedbench.algorithms.fed_tgan.fed_tgan import FedTGAN

algo = FedTGAN()
aggregator = algo.create_aggregator()
synthesizer = algo.create_synthesizer()

print(type(algo).__name__)
print(type(aggregator).__name__)
print(type(synthesizer).__name__)