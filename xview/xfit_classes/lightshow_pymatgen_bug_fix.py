from pymatgen.io import *
from pymatgen.core import Element
from tabulate import tabulate
from pymatgen.io.feff import Potential

from mp_api.client import MPRester
# from lightshow import Database
# from lightshow import _get_API_key_from_environ, pymatgen_utils


from pymatgen.io.feff.sets import FEFFDictSet




def _get_api_key(api_key):
    if api_key is None:
        api_key = _get_API_key_from_environ()
    if api_key is None:
        raise ValueError(f"Invalid API key {api_key}")
    return api_key

def _get_method(method, mpr):
    """Get all the available search methods; if the given method is not present, the search will be performed using mpr.materials.search"""
    methods = [methods for methods in dir(mpr.materials) if methods is not callable and not methods.startswith('_')]

    if method is None:
        method = None
    elif method in methods:
        method = method
    else:
        print("Searching with default method")
        method = None

    return method

# def from_materials_project(**kwargs):
#     """Constructs the :class:`.Database` object by pulling structures and
#     metadata directly from the Materials Project. This is a simple
#     passthrough method which utilizes the MPRester.materials.search\
#     API of the Materials Project v2 API.
#
#     Parameters
#     ----------
#     **kwargs
#         Description
#
#     Examples
#     --------
#
#     Deleted Parameters
#     ------------------
#     mpr_query_kwargs : dict
#         Direct passthrough to MPRester.materials.search. See
#         examples below.
#     api_key : None, optional
#         API key which can either be provided directly or is read from
#         the MP_API_KEY environment variable.
#     method : None, optional, str
#         Keyword to get different information about materials'
#         for e.g. 'thermo', 'xas', 'summary' etc. fetch information on
#         thermodynamic properties, computed XAS data, large amount of amalgated data
#         about the material, respectively. https://api.materialsproject.org/docs
#
#     Returns
#     -------
#     Database
#     """
#
#     api_key = _get_api_key(kwargs.get("api_key"))
#
#     try:
#         kwargs.pop("api_key")
#     except KeyError:
#         pass
#
#     with MPRester(api_key) as mpr:
#         method = _get_method(kwargs.get("method"), mpr=mpr)
#         try:
#             kwargs.pop("method")
#         except:
#             pass
#         if method is not None:
#             searched = getattr(mpr.materials, method).search(**kwargs)
#         else:
#             searched = mpr.materials.search(**kwargs)
#
#     structures = {s.material_id.string: s.structure if hasattr(s, "structure") else None for s in searched}
#     metadata = {s.material_id.string: s.dict() for s in searched}
#
#     # with MPRester(api_key) as mpr:
#     #     searched = mpr.materials.search(**kwargs)
#
#     # structures = {s.material_id.string: s.structure for s in searched}
#     # metadata = {s.material_id.string: s.dict() for s in searched}
#
#     return Database(structures=structures, metadata=metadata, supercells=dict())


def get_atom_map(structure=None, absorbing_atom=None):
    """
    Returns a dict that maps each atomic symbol to a unique integer starting
    from 1.

    Args:
        structure (Structure)
        absorbing_atom (str): symbol

    Returns:
        dict
    """
    unique_pot_atoms = sorted({site.specie.symbol for site in structure})

    # if there is only a single absorbing atom in the structure,
    # it should be excluded from this list
    if absorbing_atom and len(structure.indices_from_symbol(absorbing_atom)) == 1:
        unique_pot_atoms.remove(absorbing_atom)

    atom_map = {}
    for i, atom in enumerate(unique_pot_atoms):
        atom_map[atom] = i + 1
    return atom_map

class Potential_rewrite(Potential):
    def __init__(self, structure=None, absorbing_atom=None):
        super(Potential, self).__init__()
        self.struct = structure
        self.absorbing_atom = absorbing_atom
        self.pot_dict = get_atom_map(self.struct, [self.absorbing_atom])



    def __str__(self):
        """
        Returns a string representation of potential parameters to be used in
        the feff.inp file,
        determined from structure object.

                The lines are arranged as follows:

            ipot   Z   element   lmax1   lmax2   stoichiometry   spinph

        Returns:
            String representation of Atomic Coordinate Shells.
        """
        central_element = Element(self.absorbing_atom)
        ipotrow = [[0, central_element.Z, central_element.symbol, -1, -1, 0.0001, 0]]
        for el, amt in self.struct.composition.items():
            # if there is only one atom and it is the absorbing element, it should
            # be excluded from this list. Otherwise the error `No atoms or overlap
            # cards for unique pot X` will be encountered.
            # if el == central_element and amt == 1:
            #     continue
            ipot = self.pot_dict[el.symbol]
            ipotrow.append([ipot, el.Z, el.symbol, -1, -1, amt, 0])
        ipot_sorted = sorted(ipotrow, key=lambda x: x[0])
        ipotrow = str(
            tabulate(
                ipot_sorted,
                headers=[
                    "*ipot",
                    "Z",
                    "tag",
                    "lmax1",
                    "lmax2",
                    "xnatph(stoichometry)",
                    "spinph",
                ],
            )
        )
        ipotlist = ipotrow.replace("--", "**")
        return f"POTENTIALS \n{ipotlist}"


class FEFFDictSet_modified(FEFFDictSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def potential(self) -> Potential_rewrite:
        return Potential_rewrite(self.structure, self.absorbing_atom)