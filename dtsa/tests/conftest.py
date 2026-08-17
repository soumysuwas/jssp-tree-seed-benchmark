import pathlib
import sys

# dtsa/ is not an installed package. Add it, and its validation_tsp/ subfolder, to the
# import path so the DTSA tests can do flat imports (dtsa_reference, dtsa_tables, operators,
# and tsp / two_opt / tsplib_io which live under validation_tsp/).
_DTSA = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_DTSA))
sys.path.insert(0, str(_DTSA / "validation_tsp"))
