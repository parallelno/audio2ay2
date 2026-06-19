"""audio2ay2 - Convert instrumental audio into AY-3-8910 register streams.

Analysis-by-synthesis: search the legal AY register space with a cycle-accurate
emulator in the loop, scoring candidates by perceptual similarity to the source.

The AY chip is the language.
"""

__version__ = "0.1.0"
