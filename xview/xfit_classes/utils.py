# from pymatgen.io.feff.sets import MPXANESSet, MPEXAFSSet, FEFFDictSet
from xraydb import atomic_symbol
from PyQt5 import uic
import os


periodic_table = {'element' : [atomic_symbol(i) for i in range(20, 93)]}

def read_lineEdit_and_perform_sanity_check(lineEdit=None,
                                           reference_dictionary=None,
                                           message="Nothing is provided..."):
    text = lineEdit.text()
    text = text.split(" ")[0]

    if reference_dictionary is not None:
        for key, value in reference_dictionary.items():
            if text in value:
                status = True
            else:
                status = False
                print(message)
    elif text == "":
        status = False
        print(message)
    else:
        status = True

    if status:
        return text, status
    else:
        return "", status


def get_default_feff_parameters(title=None):
    cards = {"CONTROL": "1 1 1 1 1 1",
             "PRINT": "1 0 0 0 0 3",
             "RPATH": "6.0",
             "EXAFS": "20.0",
             "S02": "1.0",
             "TITLE": title}
    return cards