from typing import TYPE_CHECKING
from ...utils import _LazyModule

_import_structure = {
    "configuration_tinyaya_audio": ["TinyAyaAudioConfig", "TinyAyaAudioEncoderConfig"],
    "modeling_tinyaya_audio": [
        "TinyAyaAudioForConditionalGeneration",
        "TinyAyaAudioPreTrainedModel",
        "TinyAyaAudioEncoder",
    ],
    "processing_tinyaya_audio": ["TinyAyaAudioProcessor"],
}

if TYPE_CHECKING:
    from .configuration_tinyaya_audio import TinyAyaAudioConfig, TinyAyaAudioEncoderConfig
    from .modeling_tinyaya_audio import (
        TinyAyaAudioEncoder,
        TinyAyaAudioForConditionalGeneration,
        TinyAyaAudioPreTrainedModel,
    )
    from .processing_tinyaya_audio import TinyAyaAudioProcessor

else:
    import sys
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)