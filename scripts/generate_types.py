"""Generate TypeScript types from Pydantic domain models.

Uses datamodel-code-generator to convert JSON Schema to TypeScript interfaces.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOMAIN_PKG = ROOT / "domain"
OUTPUT_DIR = ROOT / "frontend-next" / "lib" / "generated-types"


def generate():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT))

    from domain import (
        DesignResultViewModel,
        EvidenceItem,
        HardwareInventory,
        HardwarePool,
        InferenceDesignRecommendation,
        ModelArchitecture,
        ModelIdentity,
        ValidationReport,
        WorkloadProfile,
    )
    from domain.session import DesignSession

    models = [
        ModelIdentity,
        ModelArchitecture,
        HardwareInventory,
        HardwarePool,
        EvidenceItem,
        ValidationReport,
        WorkloadProfile,
        InferenceDesignRecommendation,
        DesignResultViewModel,
        DesignSession,
    ]

    all_schemas = {}
    for model in models:
        schema = model.model_json_schema()
        all_schemas[model.__name__] = schema

    combined_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "definitions": all_schemas,
    }

    schema_path = OUTPUT_DIR / "schema.json"
    schema_path.write_text(json.dumps(combined_schema, indent=2))

    output_path = OUTPUT_DIR / "domain.ts"

    try:
        subprocess.run(
            [
                sys.executable, "-m", "datamodel_code_generator",
                "--input", str(schema_path),
                "--input-file-type", "jsonschema",
                "--output", str(output_path),
                "--output-model-type", "typescript",
                "--target-python-version", "3.11",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Generated TypeScript types at: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Type generation failed: {e.stderr}", file=sys.stderr)
        print("Falling back to JSON Schema export only.")
        _generate_manual_ts(all_schemas, output_path)
    except FileNotFoundError:
        print("datamodel-code-generator not found. Generating manual TS.")
        _generate_manual_ts(all_schemas, output_path)


def _generate_manual_ts(schemas: dict, output_path: Path):
    """Generate minimal TypeScript type exports from JSON schemas."""
    lines = [
        "// Auto-generated from Pydantic domain models",
        "// Re-generate with: make generate-types",
        "",
    ]

    for name, schema in schemas.items():
        lines.append(f"export interface {name} {{")
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        for prop_name, prop_schema in properties.items():
            ts_type = _json_type_to_ts(prop_schema)
            optional = "" if prop_name in required else "?"
            lines.append(f"  {prop_name}{optional}: {ts_type};")
        lines.append("}")
        lines.append("")

    output_path.write_text("\n".join(lines))
    print(f"Generated manual TypeScript types at: {output_path}")


def _json_type_to_ts(schema: dict) -> str:
    """Convert a JSON schema type to TypeScript."""
    if "anyOf" in schema:
        types = [_json_type_to_ts(s) for s in schema["anyOf"] if s.get("type") != "null"]
        null = any(s.get("type") == "null" for s in schema["anyOf"])
        result = " | ".join(types) if types else "unknown"
        return f"{result} | null" if null else result

    if "allOf" in schema:
        return _json_type_to_ts(schema["allOf"][0])

    if "$ref" in schema:
        ref = schema["$ref"].split("/")[-1]
        return ref

    json_type = schema.get("type", "any")

    if json_type == "string":
        if "enum" in schema:
            return " | ".join(f'"{v}"' for v in schema["enum"])
        return "string"
    elif json_type == "integer" or json_type == "number":
        return "number"
    elif json_type == "boolean":
        return "boolean"
    elif json_type == "array":
        items = schema.get("items", {})
        return f"{_json_type_to_ts(items)}[]"
    elif json_type == "object":
        if "additionalProperties" in schema:
            val_type = _json_type_to_ts(schema["additionalProperties"])
            return f"Record<string, {val_type}>"
        return "Record<string, unknown>"
    elif json_type == "null":
        return "null"

    return "unknown"


if __name__ == "__main__":
    generate()
