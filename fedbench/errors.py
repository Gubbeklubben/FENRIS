class FedBenchError(Exception):
    """Base class for exceptions raised by FedBench."""


class PluginRegistryError(FedBenchError):
    """Base for errors raised by registries."""


class DuplicateComponentError(PluginRegistryError):
    """Raised when registering an already existing component."""


class MissingComponentError(PluginRegistryError):
    """Raised if a component is missing when resolving components."""


class PluginNotFoundError(FedBenchError):
    """Raised if plugin import fails."""


class PluginRegistryNotFoundError(FedBenchError):
    """Raised if registry instance not found in plugin module. """


class InvalidPluginRegistryError(FedBenchError):
    """Raised if registry instance is invalid."""


