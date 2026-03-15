from enum import StrEnum

from flwr.common import RecordDict

from fedbench.flwr.rdict import RDictNamespaceView


class Namespace(StrEnum):
    FRAMEWORK = "framework"
    GLOBAL_INIT_ARTIFACTS = "global-init-artifacts"
    SYNTHESIZER = "synthesizer"

    def create_view(self, rdict: RecordDict) -> RDictNamespaceView:
        return RDictNamespaceView(self.value, rdict)
