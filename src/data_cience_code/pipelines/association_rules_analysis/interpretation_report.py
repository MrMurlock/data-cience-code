import os
from collections import Counter
from typing import Any

from data_cience_code.config import config
from data_cience_code.models.classification.association_rules import (
    filter_rules_by_antecedent,
    filter_rules_by_rhs,
)


def _get_reports_dir() -> str:
    out_dir = config.get_output_path("reports", "association_rules")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


_FIG_BASE = config.OUTPUT_FOLDER_ABS_PATH.replace(
    config.OUTPUT_FOLDER_ABS_PATH.split("/")[-1], "output"
)
# Build the repo-relative base for markdown image paths
# e.g., /home/.../data-cience-code/src/output/figures/association_rules/filter_by/
# becomes /src/output/figures/association_rules/filter_by/
SRC_INDEX = config.DATASET_FOLDER_ABS_PATH.find("/src/")
if SRC_INDEX == -1:
    SRC_INDEX = config.OUTPUT_FOLDER_ABS_PATH.find("/src/")
_REPO_PREFIX = config.OUTPUT_FOLDER_ABS_PATH[:SRC_INDEX] if SRC_INDEX != -1 else ""


def _fig_rel_path(abs_path: str) -> str:
    """Convert absolute figure path to repo-relative path for markdown."""
    if _REPO_PREFIX and abs_path.startswith(_REPO_PREFIX):
        return abs_path[len(_REPO_PREFIX):]
    return abs_path


_ITEM_LABELS = {
    "Gender_Female": "Female",
    "Gender_Male": "Male",
    "BMI Category_Normal": "BMI: Normal",
    "BMI Category_Obese": "BMI: Obeso",
    "BMI Category_Overweight": "BMI: Sobrepeso",
    "Sleep Disorder_Insomnia": "SD: Insomnio",
    "Sleep Disorder_None": "SD: Ninguno",
    "Sleep Disorder_Sleep Apnea": "SD: Apnea",
    "age_group_Young": "Edad: Joven",
    "age_group_Middle": "Edad: Medio",
    "age_group_Senior": "Edad: Senior",
    "quality_group_High": "Calidad: Alta",
    "quality_group_Low": "Calidad: Baja",
    "quality_group_Medium": "Calidad: Media",
    "sleep_duration_cat_Corto": "Duracion de sueño: Corto",
    "sleep_duration_cat_Medio": "Duracion de sueño: Medio",
    "sleep_duration_cat_Largo": "Duracion de sueño: Largo",
    "stress_cat_Bajo": "Estrés: Bajo",
    "stress_cat_Medio": "Estrés: Medio",
    "stress_cat_Alto": "Estrés: Alto",
    "activity_cat_Bajo": "Actividad: Baja",
    "activity_cat_Medio": "Actividad: Media",
    "activity_cat_Alto": "Actividad: Alta",
    "heart_rate_cat_Bajo": "Frecuencia cardíaca: Baja",
    "heart_rate_cat_Normal": "Frecuencia cardíaca: Normal",
    "heart_rate_cat_Alto": "Frecuencia cardíaca: Alta",
    "daily_steps_cat_Sedentario": "Cantidad de pasos: Sedentario",
    "daily_steps_cat_Moderado": "Cantidad de pasos: Moderado",
    "daily_steps_cat_Activo": "Cantidad de pasos: Activo",
    "bp_systolic_cat_Normal": "Presion Arterial Sistólica: Normal",
    "bp_systolic_cat_Elevada": "Presion Arterial Sistólica: Elevada",
    "bp_systolic_cat_Alta": "Presion Arterial Sistólica: Alta",
    "bp_diastolic_cat_Normal": "Presion Arterial Diastólica: Normal",
    "bp_diastolic_cat_Elevada": "Presion Arterial Diastólica: Elevada",
    "bp_diastolic_cat_Alta": "Presion Arterial Diastólica: Alta",
    "occupation_group_Healthcare": "Ocup: Salud",
    "occupation_group_Technical": "Ocup: Técnica",
    "occupation_group_Education": "Ocup: Educación",
    "occupation_group_Business": "Ocup: Negocios",
    "occupation_group_Sales": "Ocup: Ventas",
    "occupation_group_Legal": "Ocup: Legal",
}


def _format_item(item) -> str:
    return _ITEM_LABELS.get(item, item)


def _format_rule(row) -> str:
    ant = ", ".join(sorted(_format_item(i) for i in row["antecedents"]))
    cons = ", ".join(sorted(_format_item(i) for i in row["consequents"]))
    return f"{ant} → {cons}"


def _interpret_lift(lift: float) -> str:
    if lift > 10:
        return "extremadamente fuerte"
    if lift > 5:
        return "muy fuerte"
    if lift > 2:
        return "moderadamente fuerte"
    return "débil"


def _interpret_rule(row) -> str:
    ant_items = [_format_item(i) for i in row["antecedents"]]
    cons_items = [_format_item(i) for i in row["consequents"]]
    ant_str = ", ".join(ant_items)
    cons_str = ", ".join(cons_items)
    support = row["support"]
    confidence = row["confidence"]
    lift = row["lift"]
    leverage = row.get("leverage", 0)
    conviction = row.get("conviction", 0)
    strength = _interpret_lift(lift)
    support_pct = support * 100
    confidence_pct = confidence * 100
    lines = [
        f"- **Regla**: `{ant_str} → {cons_str}`",
        f"  - **Soporte**: {support_pct:.1f}% de los casos presentan esta combinación",
        f"  - **Confianza**: {confidence_pct:.1f}% — cuando ocurre el antecedente, "
        f"el consecuente aparece el {confidence_pct:.1f}% de las veces",
        f"  - **Lift**: {lift:.2f} — asociación {strength} "
        f"({'positiva' if lift > 1 else 'negativa'})",
    ]
    if leverage:
        lines.append(f"  - **Leverage**: {leverage:.4f} — "
                     f"diferencia entre frecuencia observada y esperada bajo independencia")
    if conviction:
        lines.append(f"  - **Convicción**: {conviction:.2f} — "
                     f"ratio de dependencia direccional")
    return "\n".join(lines)


def _variable_freq(rules) -> tuple[Counter, Counter]:
    ant_counter: Counter = Counter()
    cons_counter: Counter = Counter()
    for _, row in rules.iterrows():
        for item in row["antecedents"]:
            ant_counter[item] += 1
        for item in row["consequents"]:
            cons_counter[item] += 1
    return ant_counter, cons_counter


def _build_filter_section(
    rules,
    title: str,
    description: str,
    slug: str,
    viz_paths: dict[str, list[str]],
) -> str:
    lines = []
    a = lines.append
    a(f"## Reglas con {title}")
    a("")
    a(description)
    a("")
    a(f"Se encontraron **{len(rules)} reglas** que cumplen este criterio.\n")

    # Top 10 rules by lift
    a("### Top 10 Reglas por Lift")
    a("")
    top = rules.nlargest(10, "lift")
    for i, (_, row) in enumerate(top.iterrows(), 1):
        a(f"#### Regla {i}")
        a(_interpret_rule(row))
        a("")

    # Embedded figures
    paths = viz_paths.get(slug, [])
    for p in paths:
        fname = os.path.basename(p)
        label = fname.replace(".png", "").replace("_", " ").title()
        rel = _fig_rel_path(p)
        a(f"![{label}]({rel})")
        a("")

    return "\n".join(lines)


def _build_occupation_section(
    rules,
    slug: str,
    viz_paths: dict[str, list[str]],
) -> str:
    lines = []
    a = lines.append
    a("## Reglas con Ocupación en el Antecedente")
    a("")
    a("Esta sección analiza reglas donde el antecedente incluye el grupo ocupacional "
      "del paciente. Permite identificar qué patrones de salud, sueño y estilo de vida "
      "se asocian a cada profesión.\n")
    a(f"Se encontraron **{len(rules)} reglas** con ocupación en el antecedente.\n")

    # Occupation frequency in antecedents
    occ_items = [item for _, row in rules.iterrows() for item in row["antecedents"]
                 if "occupation_group_" in str(item)]
    occ_counter = Counter(occ_items)
    a("### Distribución de Ocupaciones en Antecedentes")
    a("")
    a("| Ocupación | Frecuencia en reglas |")
    a("|-----------|---------------------|")
    for item, freq in occ_counter.most_common():
        a(f"| {_format_item(item)} | {freq} |")
    a("")

    # Top rules
    a("### Top 10 Reglas por Lift")
    a("")
    top = rules.nlargest(10, "lift")
    for i, (_, row) in enumerate(top.iterrows(), 1):
        a(f"#### Regla {i}")
        a(_interpret_rule(row))
        a("")

    # Embedded figures
    paths = viz_paths.get(slug, [])
    for p in paths:
        fname = os.path.basename(p)
        label = fname.replace(".png", "").replace("_", " ").title()
        rel = _fig_rel_path(p)
        a(f"![{label}]({rel})")
        a("")

    return "\n".join(lines)


def _build_filtered_sections(
    rules,
    viz_paths: dict[str, list[str]],
) -> str:
    lines = []
    a = lines.append

    # Filter rules for each section
    insomnio_rules = filter_rules_by_rhs(rules, "Sleep Disorder_Insomnia")
    apnea_rules = filter_rules_by_rhs(rules, "Sleep Disorder_Sleep Apnea")
    none_rules = filter_rules_by_rhs(rules, "Sleep Disorder_None")
    ocupacion_rules = filter_rules_by_antecedent(rules, "occupation_group_")

    a(_build_filter_section(
        insomnio_rules, "Insomnio",
        "Reglas donde el consecuente es **Insomnio** (Sleep Disorder = Insomnia). "
        "Identifican los patrones de síntomas, hábitos y perfil demográfico que "
        "predicen la presencia de insomnio en los pacientes.\n",
        "insomnio", viz_paths,
    ))

    a(_build_filter_section(
        apnea_rules, "Apnea del Sueño",
        "Reglas donde el consecuente es **Apnea del Sueño** (Sleep Disorder = Sleep Apnea). "
        "Permiten identificar factores de riesgo y perfiles asociados a este trastorno "
        "respiratorio del sueño.\n",
        "apnea", viz_paths,
    ))

    a(_build_filter_section(
        none_rules, "Sin Trastorno del Sueño",
        "Reglas donde el consecuente es **Sin Trastorno** (Sleep Disorder = None). "
        "Describen los patrones asociados a la ausencia de trastornos del sueño, "
        "lo que puede orientar intervenciones preventivas.\n",
        "sin_trastorno", viz_paths,
    ))

    a(_build_occupation_section(ocupacion_rules, "ocupacion", viz_paths))

    return "\n".join(lines)


def _deduplicate_rules(rules):
    if rules is None or rules.empty:
        return rules
    return rules.drop_duplicates(
        subset=["antecedents", "consequents"]
    ).reset_index(drop=True)


def _build_report(
    results: dict[str, Any],
    viz_paths: dict[str, list[str]],
) -> str:
    rules = _deduplicate_rules(results.get("rules_apriori"))
    comparison = results.get("comparison", {})
    apriori_counts = results.get("apriori_counts", {})
    time_a = results.get("total_time_apriori", 0)
    time_f = results.get("total_time_fp", 0)

    lines = []
    a = lines.append

    a("# Reporte de Interpretación — Reglas de Asociación")
    a("")
    a("## Resumen del Análisis")
    a("")
    a(f"- **Total de reglas únicas**: {comparison.get('unique_a', 0)} (Apriori y FP-Growth "
      f"coinciden en {comparison.get('overlap_pct_a', 100)}%)")
    a(f"- **Rendimiento**: Apriori {time_a:.4f}s vs FP-Growth {time_f:.4f}s")
    supports_str = ", ".join([f"s={s}" for s in apriori_counts.keys()])
    a(f"- **Umbrales de soporte evaluados**: {supports_str}")
    a("")

    # Filtered sections (replaces old top-10-by-metric)
    if rules is not None and not rules.empty:
        a(_build_filtered_sections(rules, viz_paths))

    # Frequency of variables
    a("## Frecuencia de Variables en las Reglas")
    a("")
    a("El análisis de frecuencia de ítems en antecedentes y consecuentes permite "
      "identificar qué variables son más relevantes en las asociaciones encontradas.\n")

    if rules is not None and not rules.empty:
        ant_counter, cons_counter = _variable_freq(rules)
        top_ant = ant_counter.most_common(10)
        top_cons = cons_counter.most_common(10)

        a("### Top 10 Ítems en Antecedentes")
        a("")
        a("| Item | Frecuencia |")
        a("|------|-----------|")
        for item, freq in top_ant:
            a(f"| {_format_item(item)} | {freq} |")
        a("")
        a("### Top 10 Ítems en Consecuentes")
        a("")
        a("| Item | Frecuencia |")
        a("|------|-----------|")
        for item, freq in top_cons:
            a(f"| {_format_item(item)} | {freq} |")

    a("## Patrones por Grupo de Variable")
    a("")
    patterns = [
        ("**Presión Arterial**",
         "Las reglas con mayor lift involucran presión arterial sistólica y diastólica "
         "normales. Esto refleja la correlación natural entre ambas mediciones. "
         "Las categorías anormales (Elevada/Alta) aparecen con menos frecuencia, "
         "lo que sugiere que la hipertensión no es el estado predominante en la muestra."),
        ("**Calidad y Duración del Sueño**",
         "Aparecen reglas que asocian calidad de sueño alta con duración media/alta, "
         "y calidad baja con estrés alto. Esto valida la relación esperada entre "
         "sueño reparador y bienestar general."),
        ("**Estrés y Estilo de Vida**",
         "El estrés alto aparece frecuentemente asociado a calidad de sueño baja y "
         "actividad física baja. El estrés bajo se asocia a calidad de sueño alta. "
         "La actividad física aparece como un modulador relevante."),
        ("**Ocupación**",
         "El grupo ocupacional aparece distribuido en las reglas: profesionales de "
         "salud (Healthcare) tienden a aparecer con ciertos patrones de sueño y "
         "estrés, mientras que el grupo técnico (Technical) muestra otros."),
        ("**Trastornos del Sueño**",
         "El insomnio se asocia fuertemente con estrés alto y calidad de sueño baja. "
         "La apnea del sueño aparece más relacionada con el grupo etario Senior y "
         "categorías BMI Overweight/Obese. La ausencia de trastorno (None) se asocia "
         "a calidad de sueño alta y estrés bajo."),
    ]
    for title, desc in patterns:
        a(f"### {title}")
        a("")
        a(desc)
        a("")

    a("## Implicaciones Prácticas")
    a("")
    implications = [
        "**Detección temprana**: Las reglas con alta confianza y lift pueden utilizarse "
        "para identificar pacientes en riesgo de desarrollar trastornos del sueño. "
        "Por ejemplo, pacientes con estrés alto y calidad de sueño baja tienen alta "
        "probabilidad de presentar insomnio.",
        "**Intervenciones focalizadas**: Las asociaciones entre actividad física y "
        "calidad de sueño sugieren que promover hábitos de ejercicio podría mejorar "
        "la calidad del sueño en la población estudiada.",
        "**Perfiles de riesgo**: La combinación de variables como edad, BMI, y presión "
        "arterial permite construir perfiles de riesgo para apnea del sueño, lo que "
        "podría ayudar en screenings preventivos.",
        "**Validación con experto de dominio**: Se recomienda que un profesional de la "
        "salud valide las reglas más relevantes para confirmar su coherencia clínica "
        "y aplicabilidad en contextos reales de diagnóstico o intervención.",
    ]
    for imp in implications:
        a(f"- {imp}")
        a("")

    return "\n".join(lines)


def run(
    results: dict[str, Any],
    viz_paths: dict[str, list[str]] | None = None,
):
    print("=" * 60)
    print("  Association Rules - Interpretation Report")
    print("=" * 60)

    if viz_paths is None:
        viz_paths = {}

    out_dir = _get_reports_dir()
    report = _build_report(results, viz_paths)

    report_path = os.path.join(out_dir, "interpretation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved report to: {report_path}")
    print("Interpretation complete.\n")
