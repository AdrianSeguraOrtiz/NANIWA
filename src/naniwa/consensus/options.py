"""Select independently configurable scoring components or a named preset."""

from dataclasses import asdict, dataclass


COMPONENT_CHOICES = {
    "vote": ("all", "distinct", "families"),
    "evidence": ("none", "marginal", "joint"),
    "intensity": ("none", "global", "full"),
}


@dataclass(frozen=True)
class Components:
    vote: str
    evidence: str
    intensity: str

    def __post_init__(self):
        for name, choices in COMPONENT_CHOICES.items():
            value = getattr(self, name)
            if value not in choices:
                raise ValueError(f"Unknown {name} {value!r}. Choose one of {', '.join(choices)}")
        if (self.evidence == "none") != (self.intensity == "none"):
            raise ValueError("Use --intensity none with --evidence none, or global/full with marginal/joint evidence")


_STAGE_COMPONENTS = {
    "mean_minmax": Components("all", "none", "none"),
    "exact_mean": Components("distinct", "none", "none"),
    "family_mean": Components("families", "none", "none"),
    "marginal_global": Components("families", "marginal", "global"),
    "naniwa": Components("families", "joint", "global"),
    "joint_full": Components("families", "joint", "full"),
    "flat_joint_global": Components("distinct", "joint", "global"),
    "marginal_full": Components("families", "marginal", "full"),
}
STAGES = tuple(_STAGE_COMPONENTS)
_STAGE_NAMES = {components: stage for stage, components in _STAGE_COMPONENTS.items()}


@dataclass(frozen=True, init=False)
class ConsensusOptions:
    components: Components

    def __init__(self, stage=None, *, vote=None, evidence=None, intensity=None):
        if stage is not None and stage not in _STAGE_COMPONENTS:
            raise ValueError(f"Unknown consensus stage {stage!r}. Choose one of {', '.join(STAGES)}")
        if stage is None:
            # A vote specified on its own selects the uncorrected base vote.
            stage = "mean_minmax" if vote is not None and evidence is None and intensity is None else "naniwa"
        base = _STAGE_COMPONENTS[stage]
        if vote is None:
            vote = base.vote
        if evidence is None:
            evidence = base.evidence
            if intensity == "none":
                evidence = "none"
            elif intensity is not None and evidence == "none":
                evidence = "joint"
        if intensity is None:
            intensity = base.intensity
            if evidence == "none":
                intensity = "none"
            elif intensity == "none":
                intensity = "global"
        object.__setattr__(self, "components", Components(vote, evidence, intensity))

    @property
    def stage(self):
        """Return the registered name when a preset has these exact components."""
        return _STAGE_NAMES.get(self.components)

    def parameters(self):
        """Serialize the effective choices, including an optional registered name."""
        return {"stage": self.stage, **asdict(self.components)}

    @classmethod
    def resolve(cls, stage=None, *, vote=None, evidence=None, intensity=None):
        """Resolve a preset, component values or overrides of a preset."""
        return cls(stage, vote=vote, evidence=evidence, intensity=intensity)
