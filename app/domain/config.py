from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Tuple

Point = Tuple[float, float]


class SourceConfig(BaseModel):
    type: Literal["video_file"] = "video_file"
    path: str
    loop: bool = False


class InferenceConfig(BaseModel):
    engine: Literal["yolo"] = "yolo"
    model: str = "yolov8n.pt"
    conf: float = 0.35
    iou: float = 0.50
    classes: List[int] = Field(default_factory=lambda: [0])  # 0=person
    imgsz: int = 640


class ProcessingConfig(BaseModel):
    target_fps: float = 6.0
    resize_width: Optional[int] = 960  # None => no resize


class ZoneConfig(BaseModel):
    id: str
    polygon: List[Point]


class IntrusionRuleConfig(BaseModel):
    zone_id: str
    min_hits: int = 3
    cooldown_seconds: int = 20


class RulesConfig(BaseModel):
    intrusion: IntrusionRuleConfig


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


class AppConfig(BaseModel):
    site_id: str
    camera_id: str
    source: SourceConfig
    inference: InferenceConfig = InferenceConfig()
    processing: ProcessingConfig = ProcessingConfig()
    zones: List[ZoneConfig] = Field(default_factory=list)
    rules: RulesConfig
    evidence: EvidenceConfig = EvidenceConfig()
    storage: StorageConfig = StorageConfig()
    notify: NotifyConfig = NotifyConfig()
    ui: UiConfig = UiConfig()


    
