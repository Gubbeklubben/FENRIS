from enum import StrEnum

from flwr.app import RecordDict

from fenris.flwr.rdict import RDictNamespaceView


class Namespace(StrEnum):
    FRAMEWORK = "framework"
    GLOBAL_INIT_ARTIFACTS = "global-init-artifacts"
    SYNTHESIZER = "synthesizer"

    def view(self, rdict: RecordDict) -> RDictNamespaceView:
        return RDictNamespaceView(self.value, rdict)
