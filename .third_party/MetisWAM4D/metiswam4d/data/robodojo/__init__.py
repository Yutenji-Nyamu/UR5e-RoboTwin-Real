"""RoboDojo (RDJ_MetisWAM4D) episode data: raw windows in the RT2 contract, encoded online by ``RT2OnlineEncoder``.

``window_dataset`` also serves the other sources with the same window contract (``RDJWindowConfig.source``)."""
from metiswam4d.data.robodojo.eef_base import base_to_world_eef20, world_to_base_eef20
from metiswam4d.data.robodojo.episode_dataset import RDJEpisodeDataset, RDJWindowConfig


def window_dataset(config: RDJWindowConfig):
    if config.source == "ebench":
        from metiswam4d.data.ebench import EBenchEpisodeDataset
        return EBenchEpisodeDataset(config)
    if config.source == "vlabench":
        from metiswam4d.data.vlabench import VLABenchEpisodeDataset
        return VLABenchEpisodeDataset(config)
    if config.source == "vlabench_mix":
        from metiswam4d.data.vlabench.official import VLABenchMixDataset
        return VLABenchMixDataset(config)
    if config.source != "rdj":
        raise ValueError(f"unknown window source {config.source!r}")
    return RDJEpisodeDataset(config)


__all__ = ["RDJEpisodeDataset", "RDJWindowConfig", "base_to_world_eef20", "window_dataset", "world_to_base_eef20"]
