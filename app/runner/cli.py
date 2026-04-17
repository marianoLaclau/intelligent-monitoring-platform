# runner/cli.py
"""
Entrypoint CLI: carga YAML → construye componentes → ejecuta pipeline.
"""
import os
import yaml
from loguru import logger

from app.domain.config import AppConfig
from app.adapters.sources.video_file import VideoFileSource
from app.core.inference_yolo import YoloInferencer
from app.core.zones import ZoneIndex
from app.core.tracking import TrackStateManager
from app.core.rules import RuleEngine, IntrusionRule
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
    logger.info(f"Tracking: {'ENABLED' if cfg.tracking.enabled else 'DISABLED'}")

    # ── Source ──
    source = VideoFileSource(path=src_path, loop=cfg.source.loop)

    # ── Inferencer (con o sin tracking integrado) ──
    tracker_config = None
    if cfg.tracking.enabled:
        tracker_config = {
            "track_high_thresh": cfg.tracking.track_high_thresh,
            "track_low_thresh": cfg.tracking.track_low_thresh,
            "new_track_thresh": cfg.tracking.new_track_thresh,
            "track_buffer": cfg.tracking.track_buffer,
            "match_thresh": cfg.tracking.match_thresh,
            # Re-ID (BoT-SORT)
            "with_reid": cfg.tracking.with_reid,
            "appearance_thresh": cfg.tracking.appearance_thresh,
            "proximity_thresh": cfg.tracking.proximity_thresh,
        }

    inferencer = YoloInferencer(
        model_path=cfg.inference.model,
        conf=cfg.inference.conf,
        iou=cfg.inference.iou,
        classes=cfg.inference.classes,
        imgsz=cfg.inference.imgsz,
        tracking_enabled=cfg.tracking.enabled,
        tracker_type=cfg.tracking.tracker,
        tracker_config=tracker_config,
    )

    # ── Zones ──
    zones_data = [
        {
            "id": z.id,
            "polygon": z.polygon,
            "zone_type": z.zone_type,
        }
        for z in cfg.zones
        if z.active
    ]
    zone_index = ZoneIndex(zones_data)
    logger.info(f"Zones loaded: {zone_index.zone_ids}")

    # ── Tracking / Rules ──
    track_manager = None
    rule_engine = None
    intrusion_rule_legacy = None

    if cfg.tracking.enabled:
        track_manager = TrackStateManager(
            min_hits_enter=cfg.tracking.min_hits_enter,
            max_misses_exit=cfg.tracking.max_misses_exit,
            track_timeout_seconds=cfg.tracking.track_timeout_seconds,
            dwell_alert_seconds=cfg.tracking.dwell_alert_seconds,
        )

        # Construir reglas por zona
        zone_rules = {}
        for zr in cfg.rules.get_zone_rules():
            zone_rules[zr.zone_id] = {
                "cooldown_seconds": zr.cooldown_seconds,
                "dwell_alert_seconds": zr.dwell_alert_seconds,
            }
            # Ajustar cooldown por track en el TrackStateManager
            track_manager.cooldown_per_track_seconds = zr.cooldown_seconds

        rule_engine = RuleEngine(zone_rules=zone_rules)
        logger.info(f"TrackStateManager configured: min_hits={cfg.tracking.min_hits_enter} "
                     f"max_misses={cfg.tracking.max_misses_exit}")
    else:
        # Legacy mode
        zone_rules_list = cfg.rules.get_zone_rules()
        if zone_rules_list:
            zr = zone_rules_list[0]
            intrusion_rule_legacy = IntrusionRule(
                zone_id=zr.zone_id,
                min_hits=zr.min_hits,
                cooldown_seconds=zr.cooldown_seconds,
            )
            logger.info(f"Legacy IntrusionRule: zone={zr.zone_id}")

    # ── Adapters ──
    snapshot_writer = SnapshotWriter(snap_dir) if cfg.evidence.snapshot.enabled else None
    store = JsonlEventStore(jsonl_path)

    notifier = None
    if cfg.notify.webhook.enabled:
        if not cfg.notify.webhook.url:
            raise RuntimeError("notify.webhook.enabled=true pero falta notify.webhook.url")
        notifier = WebhookNotifier(cfg.notify.webhook.url, cfg.notify.webhook.timeout_seconds)

    # ── Pipeline ──
    pipe = Pipeline(
        site_id=cfg.site_id,
        camera_id=cfg.camera_id,
        source=source,
        inferencer=inferencer,
        zone_index=zone_index,
        track_manager=track_manager,
        rule_engine=rule_engine,
        intrusion_rule=intrusion_rule_legacy,
        snapshot_writer=snapshot_writer,
        event_store=store,
        notifier=notifier,
        processing_cfg=cfg.processing,
        ui_cfg=cfg.ui,
    )

    pipe.run()
