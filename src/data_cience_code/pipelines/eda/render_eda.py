"""Render de los documentos del EDA: reporte y presentación Marp.

Separa el render del análisis (docs/1.EDA.md §11): consume los artifacts ya
existentes y resuelve las directivas {{ table()/metric()/image() }} de los
templates Markdown de `reports/eda/`. Los documentos finales se guardan en
`output/eda/`, separando los documentos editables de los outputs generados.
No recalcula estadísticas ni regenera imágenes.

* `render_reports()`: regenera EDA.final.md y presentation.final.md.
* `export_pptx()`: opcional, invoca Marp CLI si está disponible y convierte
  la presentación renderizada a PPTX.
"""
import shutil
import subprocess
from pathlib import Path

from data_cience_code.utils.render import Renderer

from .config_eda import ARTIFACTS_DIR, OUTPUTS_EDA_DIR


REPORT_TEMPLATE = Path("reports/eda/EDA.md")
PRESENTATION_TEMPLATE = Path("reports/eda/presentation.md")

REPORT_OUTPUT = OUTPUTS_EDA_DIR / "EDA.final.md"
PRESENTATION_OUTPUT = OUTPUTS_EDA_DIR / "presentation.final.md"
PRESENTATION_PPTX = OUTPUTS_EDA_DIR / "presentation.final.pptx"

_DOCUMENTS = (
    (REPORT_TEMPLATE, REPORT_OUTPUT),
    (PRESENTATION_TEMPLATE, PRESENTATION_OUTPUT),
)


def _renderer() -> Renderer:
    return Renderer(
        artifacts_base=ARTIFACTS_DIR,
        tables_dir=ARTIFACTS_DIR / "tables",
        metrics_dir=ARTIFACTS_DIR / "metrics",
        images_dir=ARTIFACTS_DIR / "images",
    )


def render_reports() -> list[Path]:
    """Renderiza reporte y presentación a partir de los artifacts existentes."""
    renderer = _renderer()
    outputs = []
    for template_path, output_path in _DOCUMENTS:
        if not template_path.exists():
            raise FileNotFoundError(f"Missing template {template_path}")
        rendered = renderer.render_file(template_path, output_path)
        outputs.append(rendered)
        print(f"Rendered {template_path} -> {rendered}")

    missing = renderer.last_missing
    if missing:
        print(f"Renderer warning: unresolved artifacts -> {', '.join(missing)}")
    return outputs


def export_pptx() -> Path | None:
    """Exporta la presentación a PPTX con Marp CLI (si está instalada)."""
    if shutil.which("marp") is None:
        print("marp CLI no disponible; se omite la exportación a PPTX")
        return None
    PRESENTATION_PPTX.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["marp", str(PRESENTATION_OUTPUT), "--pptx", "--allow-local-files", "-o", str(PRESENTATION_PPTX)],
        check=True,
    )
    print(f"Wrote {PRESENTATION_PPTX}")
    return PRESENTATION_PPTX
