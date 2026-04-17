# tests/conftest.py
"""Configuración global de pytest."""
import sys
import os

# Asegurar que el directorio raíz del proyecto está en el path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
