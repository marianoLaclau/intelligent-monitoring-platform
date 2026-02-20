import os
import yaml
from loguru import logger
from app.domain.config import AppConfig
from app.adapters.sources.video_file import VideoFileSource
from app.core.inference_yolo import YoloPersonInferencer
from app.core.zones import ZoneIndex
from app.core.rules import IntrusionRule
from app.adapters.evidence.snapshot import SnapshotWriter
from app.adapters.storage.fs_jsonl import JsonlEventStore
from app.adapters.notify.webhook import WebhookNotifier
from app.core.pipeline import Pipeline


def _resolve_path(base_dir: str, p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base_dir, p))


def run_from_yaml(config_path: str) -> None:
    config_path = os.path.normpath(config_path)
    base_dir = os.path.dirname(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = AppConfig.model_validate(raw)

    # Resolver paths relativos al YAML
    src_path = _resolve_path(base_dir, cfg.source.path)
    jsonl_path = _resolve_path(base_dir, cfg.storage.jsonl.path)
    snap_dir = _resolve_path(base_dir, cfg.evidence.snapshot.out_dir)

    logger.info(f"Config OK. YAML base_dir={base_dir}")
    logger.info(f"Video source path={src_path}")

    source = VideoFileSource(path=src_path, loop=cfg.source.loop)

    inferencer = YoloPersonInferencer(
        model_path=cfg.inference.model,
        conf=cfg.inference.conf,
        iou=cfg.inference.iou,
        classes=cfg.inference.classes,
        imgsz=cfg.inference.imgsz,
    )

    zones = [{"id": z.id, "polygon": z.polygon} for z in cfg.zones]
    zone_index = ZoneIndex(zones)

    intrusion_rule = IntrusionRule(
        zone_id=cfg.rules.intrusion.zone_id,
        min_hits=cfg.rules.intrusion.min_hits,
        cooldown_seconds=cfg.rules.intrusion.cooldown_seconds,
    )

    snapshot_writer = SnapshotWriter(snap_dir) if cfg.evidence.snapshot.enabled else None
    store = JsonlEventStore(jsonl_path)

    notifier = None
    if cfg.notify.webhook.enabled:
        if not cfg.notify.webhook.url:
            raise RuntimeError("notify.webhook.enabled=true pero falta notify.webhook.url")
        notifier = WebhookNotifier(cfg.notify.webhook.url, cfg.notify.webhook.timeout_seconds)

    pipe = Pipeline(
        site_id=cfg.site_id,
        camera_id=cfg.camera_id,
        source=source,
        inferencer=inferencer,
        zone_index=zone_index,
        intrusion_rule=intrusion_rule,
        snapshot_writer=snapshot_writer,
        event_store=store,
        notifier=notifier,
        processing_cfg=cfg.processing,
        ui_cfg=cfg.ui,
    )

    pipe.run()
