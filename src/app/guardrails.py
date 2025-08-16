from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml
import re
from typing import List

@dataclass(frozen=True)
class Catalog:
    diplomas: List[str]

def load_catalog(path: str = "data/catalog.yaml") -> Catalog | None:
    p = Path(path)
    if not p.exists():
        return None
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return Catalog(diplomas=[str(x).strip() for x in data.get("diplomas", [])])

def looks_like_diploma_query(text: str) -> bool:
    t = text.strip()
    # كلمات دالة. عدّلها بحرّية
    return bool(re.search(r"\b(دبلوم|الدبلومات|دبلومات)\b", t))

def answer_diplomas(catalog: Catalog) -> str:
    if not catalog or not catalog.diplomas:
        return "لا توجد قائمة دبلومات مُحدّثة في النظام حاليًا."
    items = "\n".join(f"- دبلوم {name}" for name in catalog.diplomas)
    return f"""الدبلومات المعتمدة حاليًا في المعهد هي:
{items}
"""
