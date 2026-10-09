"""
Color definitions and complementary mappings for binding sites.

The color system defines binding complementarity: each color binds only to its
complement. This creates a bipartite-like binding structure where:
- white ↔ black
- red ↔ green  
- blue ↔ yellow

This is modeled after DNA base pairing (A↔T, C↔G) but with 3 complementary pairs.
"""

from enum import Enum, auto
from typing import Dict, Set, FrozenSet
from dataclasses import dataclass


class Color(Enum):
    """Binding site colors with explicit complementarity."""
    WHITE = auto()
    BLACK = auto()
    RED = auto()
    GREEN = auto()
    BLUE = auto()
    YELLOW = auto()
    
    def __str__(self) -> str:
        return self.name.lower()
    
    @property
    def complement(self) -> 'Color':
        """Return the complementary binding color."""
        return COMPLEMENT_MAP[self]


# Complement mapping (bidirectional)
COMPLEMENT_MAP: Dict[Color, Color] = {
    Color.WHITE: Color.BLACK,
    Color.BLACK: Color.WHITE,
    Color.RED: Color.GREEN,
    Color.GREEN: Color.RED,
    Color.BLUE: Color.YELLOW,
    Color.YELLOW: Color.BLUE,
}

# String representation for serialization
COLOR_STR_MAP: Dict[str, Color] = {c.name.lower(): c for c in Color}
# Add extra mappings for DNA bases
COLOR_STR_MAP.update({
    'a': Color.WHITE,
    't': Color.BLACK,
    'c': Color.RED,
    'g': Color.GREEN,
})
COMPLEMENTS_STR: Dict[str, str] = {str(k): str(v) for k, v in COMPLEMENT_MAP.items()}


@dataclass(frozen=True)
class ColorPalette:
    """
    Immutable color palette defining the binding alphabet.
    
    The palette defines the complete set of colors available in the system
    and their complementarity relationships.
    """
    colors: FrozenSet[Color]
    complement_map: Dict[Color, Color]
    
    def __post_init__(self):
        # Validate complementarity is symmetric and complete
        for c, comp in self.complement_map.items():
            assert self.complement_map[comp] == c, f"Complementarity not symmetric for {c}"
            assert c in self.colors and comp in self.colors, f"Color {c} or {comp} not in palette"
    
    @classmethod
    def standard(cls) -> 'ColorPalette':
        """Create the standard 6-color, 3-pair palette."""
        colors = frozenset(Color)
        return cls(colors=colors, complement_map=COMPLEMENT_MAP)
    
    @classmethod
    def from_strings(cls, color_pairs: Dict[str, str]) -> 'ColorPalette':
        """Create palette from string color pairs."""
        colors = set()
        complement_map = {}
        for c1_str, c2_str in color_pairs.items():
            c1 = COLOR_STR_MAP[c1_str.lower()]
            c2 = COLOR_STR_MAP[c2_str.lower()]
            colors.add(c1)
            colors.add(c2)
            complement_map[c1] = c2
            complement_map[c2] = c1
        return cls(colors=frozenset(colors), complement_map=complement_map)
    
    def get_channels(self) -> list[tuple[Color, Color]]:
        """Get unique binding channels (color, complement) without duplication."""
        seen = set()
        channels = []
        for c in self.colors:
            comp = self.complement_map[c]
            # Use sorted tuple as canonical key
            pair = tuple(sorted([c, comp], key=lambda x: x.value))
            if pair not in seen:
                seen.add(pair)
                # Always use the sorted pair as the channel
                channels.append(pair)
        return channels
    
    def __contains__(self, color: Color) -> bool:
        return color in self.colors
    
    def __len__(self) -> int:
        return len(self.colors)


# Default standard palette
DEFAULT_PALETTE = ColorPalette.standard()

# Backward compatibility with original string-based COMPLEMENTS
COMPLEMENTS = COMPLEMENTS_STR