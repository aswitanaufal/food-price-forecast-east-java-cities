import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "ingest_data.py"


spec = importlib.util.spec_from_file_location("ingest_data", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_raw_dirs_resolve_to_project_data_raw():
    expected_root = ROOT
    expected_raw = expected_root / "project" / "data" / "raw"

    assert module.BASE_DIR == expected_root
    assert module.RAW_DIR == expected_raw
    assert module.PIHPS_RAW_DIR == expected_raw / "pihps"
    assert module.OPEN_METEO_RAW_DIR == expected_raw / "open_meteo"
