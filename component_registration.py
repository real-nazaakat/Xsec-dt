
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Iterable, List


@dataclass
class Component:
    component_id: str
    role: str
    os_type: str
    layer: str
    criticality: int        # 1..5
    exposure_level: str
    is_patched: bool

    def to_dict(self) -> Dict:
        return asdict(self)


class ComponentRegistry:


    def __init__(self):
        self._store: Dict[str, Component] = {}

    def register(self, component: Component) -> None:
        self._store[component.component_id] = component

    def get(self, component_id: str) -> Optional[Component]:
        return self._store.get(component_id)

    def list_components(self) -> List[Component]:
        return list(self._store.values())

    @classmethod
    def from_rows(cls, rows: Iterable[Iterable[str]]) -> "ComponentRegistry":

        it = iter(rows)
        header = [h.strip() for h in next(it)]
        idx = {name: i for i, name in enumerate(header)}

        reg = cls()
        for r in it:
            if len(r) < len(header):
                continue

            comp = Component(
                component_id=r[idx["component_id"]].strip(),
                role=r[idx["role"]].strip(),
                os_type=r[idx["os_type"]].strip(),
                layer=r[idx["layer"]].strip(),
                criticality=int(r[idx["criticality"]]),
                exposure_level=r[idx["exposure_level"]].strip(),
                is_patched=r[idx["is_patched"]].strip().lower() in {"1", "true", "yes", "y", "t"},
            )
            reg.register(comp)

        return reg
