"""Import placeholder on the VLABench evaluation client's path: ``VLABench.evaluation`` imports its OpenVLA policy
wrapper (``from transformers import ...``) at package import; the OpenWAM client never instantiates it."""


class _Unavailable:
    def __init__(self, *args, **kwargs):
        raise ImportError("transformers is not installed in the VLABench client environment")

    from_pretrained = classmethod(__init__)


AutoModelForVision2Seq = AutoProcessor = _Unavailable
