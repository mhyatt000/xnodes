from __future__ import annotations

from pathlib import Path

BASE = Path(__file__).parent.parent
HOME = Path.home()
DATA = HOME / "data_xgym"

# the naming convention is as follows:
# 0_input: raw DATA
# 1_clip: already been clipped DATA
# 2_hamer: already been hamer(ed) DATA
# 3_center: already been centered DATA
# 4_clean: already been cleaned DATA

MANO = DATA / "mano"
MANO_0 = MANO / "0_input"
MANO_1 = MANO / "1_clip"
MANO_2 = MANO / "2_hamer"
MANO_3 = MANO / "3_center"
MANO_4 = MANO / "4_clean"
# filtered so hands are in frame
# clipped
# cleaned for noop

MANO_0DONE = MANO_0.parent / f"{MANO_0.stem}_done"
MANO_1DONE = MANO_1.parent / f"{MANO_1.stem}_done"
MANO_2DONE = MANO_2.parent / f"{MANO_2.stem}_done"
MANO_3DONE = MANO_3.parent / f"{MANO_3.stem}_done"

for d in [DATA, MANO, MANO_0, MANO_1, MANO_2, MANO_3, MANO_4]:
    d.mkdir(exist_ok=True)
for d in [MANO_0DONE, MANO_1DONE, MANO_2DONE, MANO_3DONE]:
    d.mkdir(exist_ok=True)
