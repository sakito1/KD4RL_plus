import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_experiments.run_paper_experiments_final import format_scaled


def test_format_scaled_uses_fixed_decimal_places():
    assert format_scaled(2.655282, scale=100.0, suffix="%") == "265.53%"
    assert format_scaled(1.152742, scale=1.0) == "1.15"
    assert format_scaled(0.03304, scale=100.0, suffix=" pp", signed=True) == "+3.30 pp"


if __name__ == "__main__":
    test_format_scaled_uses_fixed_decimal_places()
