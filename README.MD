intelligent-monitoring-platform/
  app/
    __init__.py

    domain/                 # ENTIDADES + ESQUEMAS (puros)
      models.py             # dataclasses / pydantic base
      events.py             # EventType, EventPayload
      config.py             # esquema de configuración

    core/                   # MOTOR (sin dependencias web)
      pipeline.py           # orquestador del flujo
      inference_yolo.py     # implementación YOLO
      tracking.py           # tracker (simple ahora)
      zones.py              # shapely point-in-polygon
      rules.py              # intrusión + cooldown + persistencia
      clock.py              # timestamps consistentes

    adapters/               # INTEGRACIONES (enchufables)
      sources/
        video_file.py       # fuente desde archivo
        rtsp.py             # (stub para después)
      evidence/
        snapshot.py         # guardar imagen
        clip_buffer.py      # buffer circular (stub MVP->simple)
      notify/
        webhook.py          # n8n webhook
      storage/
        fs_jsonl.py         # log de eventos en JSONL

    runner/                 # ENTRYPOINTS
      cli.py                # corre un sitio/cámara desde YAML
      __main__.py           # python -m app.runner

    api/                    # (NO USAR AHORA, solo stub)
      main.py               # FastAPI app (más adelante)
      routes.py

  config/
    sites/
      demo.yaml
    default.yaml

  outputs/
    evidence/.gitkeep
    events/.gitkeep
    logs/.gitkeep

  tests/
    test_zones.py
    test_rules.py


