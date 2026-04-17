# domain/config.py
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Tuple

Point = Tuple[float, float]


class SourceConfig(BaseModel):
    type: Literal["video_file", "rtsp"] = "video_file"
    path: str
    loop: bool = False


class InferenceConfig(BaseModel):
    engine: Literal["yolo"] = "yolo"
    model: str = "yolov8n.pt"
    conf: float = 0.25              # bajo para no perder detecciones intermitentes
    iou: float = 0.50
    classes: List[int] = Field(default_factory=lambda: [0])
    imgsz: int = 640


class TrackingConfig(BaseModel):
    """
    Parámetros de BoT-SORT/ByteTrack.
    Los defaults son los de Ultralytics — no tocar salvo necesidad real.
    Solo cambiamos: with_reid=True y track_buffer más alto.
    """
    enabled: bool = True
    tracker: Literal["botsort", "bytetrack"] = "botsort"

    # Defaults de Ultralytics (probados y calibrados por ellos)
    track_high_thresh: float = 0.3
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.4
    track_buffer: int = 150         # default=30, subimos para dar tiempo al ReID
    match_thresh: float = 0.8

    # ReID
    with_reid: bool = True
    appearance_thresh: float = 0.25
    proximity_thresh: float = 0.5

    # Nuestro TrackStateManager
    min_hits_enter: int = 3
    max_misses_exit: int = 15
    track_timeout_seconds: float = 30.0
    dwell_alert_seconds: float = 60.0


class ProcessingConfig(BaseModel):
    target_fps: float = 6.0
    resize_width: Optional[int] = 960


class ZoneConfig(BaseModel):
    id: str
    polygon: List[Point]
    zone_type: Literal["restricted", "monitored", "counting"] = "restricted"
    active: bool = True
    schedule: Optional[str] = None


class IntrusionRuleConfig(BaseModel):
    zone_id: str
    min_hits: int = 3
    cooldown_seconds: int = 20
    dwell_alert_seconds: float = 60.0


class RulesConfig(BaseModel):
    intrusion: Optional[IntrusionRuleConfig] = None
    intrusion_zones: List[IntrusionRuleConfig] = Field(default_factory=list)

    def get_zone_rules(self) -> List[IntrusionRuleConfig]:
        rules = list(self.intrusion_zones)
        if self.intrusion is not None:
            existing_ids = {r.zone_id for r in rules}
            if self.intrusion.zone_id not in existing_ids:
                rules.append(self.intrusion)
        return rules


class SnapshotConfig(BaseModel):
    enabled: bool = True
    out_dir: str = "outputs/snapshots"


class EvidenceConfig(BaseModel):
    snapshot: SnapshotConfig = SnapshotConfig()


class JsonlStorageConfig(BaseModel):
    path: str = "outputs/events.jsonl"


class StorageConfig(BaseModel):
    jsonl: JsonlStorageConfig = JsonlStorageConfig()


class WebhookConfig(BaseModel):
    enabled: bool = False
    url: Optional[str] = None
    timeout_seconds: int = 5


class NotifyConfig(BaseModel):
    webhook: WebhookConfig = WebhookConfig()


class UiConfig(BaseModel):
    show_window: bool = True
    window_name: str = "MVP Monitor"
    infer_every_n_frames: int = 2   # cada N frames se ejecuta YOLO+tracker
    show_track_id: bool = True
    show_track_trail: bool = True
    trail_length: int = 30


class AppConfig(BaseModel):
    site_id: str
    camera_id: str
    source: SourceConfig
    inference: InferenceConfig = InferenceConfig()
    tracking: TrackingConfig = TrackingConfig()
    processing: ProcessingConfig = ProcessingConfig()
    zones: List[ZoneConfig] = Field(default_factory=list)
    rules: RulesConfig
    evidence: EvidenceConfig = EvidenceConfig()
    storage: StorageConfig = StorageConfig()
    notify: NotifyConfig = NotifyConfig()
    ui: UiConfig = UiConfig()
