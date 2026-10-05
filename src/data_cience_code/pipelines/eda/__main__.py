"""Entrypoint del pipeline EDA.

Uso:
    uv run python -m data_cience_code.pipelines.eda                 # análisis completo + render
    uv run python -m data_cience_code.pipelines.eda --render-only   # solo render (artifacts existentes)
    uv run python -m data_cience_code.pipelines.eda --no-render     # análisis sin render de documentos
    uv run python -m data_cience_code.pipelines.eda --pptx          # además exporta la presentación a PPTX
"""
import argparse

from .pipeline import main


def run(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="data_cience_code.pipelines.eda",
        description=(
            "EDA: preprocessing, calidad, análisis univariado y bivariado, "
            "generación de artifacts y render de reporte/presentación."
        ),
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="solo render de reporte/presentación desde artifacts existentes (sin re-análisis)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="ejecutar el análisis sin render de documentos",
    )
    parser.add_argument(
        "--pptx",
        action="store_true",
        help="exportar la presentación a PPTX con Marp CLI (si está instalada)",
    )
    args = parser.parse_args(argv)

    if args.render_only:
        from .render_eda import export_pptx, render_reports

        render_reports()
        if args.pptx:
            export_pptx()
        return

    main(render_documents=not args.no_render, pptx=args.pptx)


if __name__ == "__main__":
    run()
