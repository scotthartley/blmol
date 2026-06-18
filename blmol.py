"""Scripts for loading molecular geometries into Blender.

    Main use is to define the "molecule" object, which can be used to
    draw models of molecular coordinates.

    Written by Scott Hartley, www.hartleygroup.org and
    blog.hartleygroup.org.

    Hosted here:

    https://github.com/scotthartley/blmol

    Some aspects (especially rotating the bond cylinders) based on this
    wonderful blog post from Patrick Fuller:

    http://patrick-fuller.com/molecules-from-smiles-molfiles-in-blender/

    His project is hosted here:

    https://github.com/patrickfuller/blender-chemicals
"""

import numpy as np
import time

# Python outside of Blender doesn't play all that well with bpy, so need
# to handle ImportError.
try:
    import bpy
    import bmesh
    from mathutils import Matrix, Vector
    bpy_avail = True
except ImportError:
    bpy_avail = False

# Dictionary of color definitions (RGB + alpha tuples). Blender RGB
# colors can be conveniently determined by using a uniform png of the
# desired color as a background image, then using the eyedropper. Colors
# with the "_cb" tag are based on colorblind-safe colors as described in
# this Nature Methods editorial DOI:10.1038/nmeth.1618.
COLORS = {
    'black': (0, 0, 0, 1.0),
    'dark_red': (0.55, 0, 0, 1.0),
    'gray': (0.2, 0.2, 0.2, 1.0),
    'dark_gray': (0.1, 0.1, 0.1, 1.0),
    'green': (0.133, 0.545, 0.133, 1.0),
    'dark_green': (0.1, 0.5, 0.1, 1.0),
    'indigo': (0.294, 0, 0.509, 1.0),
    'light_gray': (0.7, 0.7, 0.7, 1.0),
    'orange': (1.0, 0.647, 0, 1.0),
    'purple': (0.627, 0.125, 0.941, 1.0),
    'red': (0.8, 0, 0, 1.0),
    'royal_blue': (0.255, 0.412, 0.882, 1.0),
    'white': (1.0, 1.0, 1.0, 1.0),
    'yellow': (1.0, 1.0, 0, 1.0),
    'violet': (0.561, 0, 1.0, 1.0),
    'blue_cb': (0, 0.168, 0.445, 1.0),
    'bluish_green_cb': (0, 0.620, 0.451, 1.0),
    'orange_cb': (0.791, 0.347, 0, 1.0),
    'reddish_purple_cb': (0.800, 0.475, 0.655, 1.0),
    'sky_blue_cb': (0.337, 0.706, 0.914, 1.0),
    'vermillion_cb': (0.665, 0.112, 0, 1.0),
    'yellow_cb': (0.871, 0.776, 0.054, 1.0),
    }

ATOMIC_NUMBERS = {
    'H': 1,
    'LI': 3,
    'B': 5,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
    'NA': 11,
    'MG': 12,
    'P': 15,
    'S': 16,
    'CL': 17,
    'K': 19,
    'ZN': 30,
    'BR': 35,
    'RB': 37,
    'I': 53,
    'CS': 55
    }

# Dictionary of Van der Waals radii, by atomic number, from Wolfram
# Alpha.
RADII = {
    1: 1.20,
    3: 1.82,
    5: 1.92,  # From wikipedia
    6: 1.70,
    7: 1.55,
    8: 1.52,
    9: 1.47,
    11: 2.27,
    12: 1.73,  # From wikipedia
    15: 1.80,
    16: 1.80,
    17: 1.75,
    19: 2.75,
    30: 1.39,  # From wikipedia
    35: 1.85,
    37: 3.03,
    53: 1.98,
    55: 3.43
    }

# Dictionaries of colors for drawing elements, by atomic number. Used by
# several functions when the `color = 'by_element'` option is passed.
ELEMENT_COLORS = {
    1: 'white',
    3: 'violet',
    5: 'orange',
    6: 'gray',
    7: 'royal_blue',
    8: 'red',
    9: 'green',
    11: 'violet',
    12: 'dark_green',
    15: 'orange',
    16: 'yellow',
    17: 'green',
    19: 'violet',
    30: 'dark_gray',
    35: 'dark_red',
    37: 'violet',
    53: 'indigo',
    55: 'violet'
    }

# Conversion factors for 1 BU. Default is typically 1 nm. Assumes
# geometries are input with coords in angstroms.
UNIT_CONV = {
    'nm': 0.1,
    'A': 1.0
    }


def _create_new_material(name, color):
    """Create a new material.

    Args:
        name (str): Name for the new material (e.g., 'red')
        color (tuple): RGB color for the new material (diffuse_color)
            (e.g., (1, 0, 0, 1))
    Returns:
        The new material.
    """

    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.roughness = 0.5
    mat.specular_color = (1, 1, 1)
    mat.specular_intensity = 0.2

    return mat


class Atom:
    """A single atom.

    Attributes:
        at_num (int): The atomic number of the atom.
        location (numpy array): The xyz location of the atom, in
            Angstroms.
        id_num (int): A unique identifier number.
    """

    def __init__(self, atomic_number, location, id_num):
        self.at_num = atomic_number
        self.location = location  # np.array
        self.id_num = id_num

    def draw(self, color='by_element', radius=None, units='nm',
             scale=1.0, subsurf_level=2, segments=16, mesh_cache=None):
        """Draw the atom in Blender.

        Args:
            color (string, ='by_element'): If 'by_element', coloring
                uses ELEMENT_COLORS. Otherwise specifies the color
                (must be a key in COLORS).
            radius (string, =None): If None, draws at the van der Waals
                radius. Otherwise specifies the radius in angstroms.
            units (sting, ='nm'): 1 BU = 1 nm by default. Can also be
                set to angstroms.
            scale (float, =1.0): Scaling factor for the atom. Useful
                when generating ball-and-stick models. Applied via
                object scale so the underlying mesh can be shared.
            subsurf_level (int, =2): Subsurface subdivisions that will
                be applied.
            segments (int, =16): Number of segments in each UV sphere
                primitive
            mesh_cache (dict, =None): Optional shared dict of
                {(at_num, rad_adj, segments): bpy.types.Mesh} used to
                reuse one mesh across atoms with matching geometry.
                Pass an empty dict to enable instancing.

        Returns:
            The blender object.
        """

        # The corrected location (i.e., scaled to units.)
        loc_corr = tuple(c*UNIT_CONV[units] for c in self.location)

        # Sphere radius in BU. Scale is applied to the object below so
        # the mesh stays canonical and can be shared across atoms.
        if radius is None:
            rad_adj = RADII[self.at_num]*UNIT_CONV[units]
        else:
            rad_adj = radius*UNIT_CONV[units]

        mesh_key = (self.at_num, rad_adj, segments)
        me = mesh_cache.get(mesh_key) if mesh_cache is not None else None
        new_mesh = me is None
        if new_mesh:
            bm = bmesh.new()
            bmesh.ops.create_uvsphere(bm,
                                      u_segments=segments,
                                      v_segments=segments,
                                      radius=rad_adj)
            for f in bm.faces:
                f.smooth = True

            me = bpy.data.meshes.new("atom_mesh({})".format(self.at_num))
            bm.to_mesh(me)
            bm.free()
            if mesh_cache is not None:
                mesh_cache[mesh_key] = me

        # Assign mesh to object and place in space.
        atom_sphere = bpy.data.objects.new("atom({})_{}".format(
                            self.at_num, self.id_num), me)
        bpy.context.collection.objects.link(atom_sphere)

        atom_sphere.location = loc_corr
        if scale != 1.0:
            atom_sphere.scale = (scale, scale, scale)

        # Assign subsurface modifier, if requested
        if subsurf_level != 0:
            atom_sphere.modifiers.new('Subsurf', 'SUBSURF')
            atom_sphere.modifiers['Subsurf'].levels = subsurf_level

        # Color atom and assign material. Material lives on the mesh,
        # so only append when the mesh is freshly built; reused meshes
        # already carry the material from the first atom.
        if new_mesh:
            if color == 'by_element':
                atom_color = ELEMENT_COLORS[self.at_num]
            else:
                atom_color = color

            if atom_color not in bpy.data.materials:
                _create_new_material(atom_color, COLORS[atom_color])

            me.materials.append(bpy.data.materials[atom_color])

        return atom_sphere


class Bond:
    """A bond between two atoms.

    Attributes:
        atom1 (atom): The first atom in the bond.
        atom2 (atom): The second atom in the bond.
    """

    def __init__(self, atom1, atom2):
        self.atom1 = atom1
        self.atom2 = atom2

    @staticmethod
    def _draw_half(location, length, rot_angle, rot_axis, element,
                   radius=0.2, color='by_element', units='nm',
                   vertices=64, edge_split=False):
        """Draw half of a bond (static method).

        Draws half of a bond, given the location and length. Bonds are
        drawn in halves to facilitate coloring by element.

        Args:
            location (np.array): The center point of the half bond.
            length (float): The length of the half bond.
            rot_angle (float): Angle by which bond will be rotated.
            rot_axis (np.array): Axis of rotation.
            element (int): atomic number of element of the bond (for
                coloring).
            radius (float, =0.2): radius of the bond.
            color (string, ='by_element'): color of the bond. If
                'by_element', uses element coloring.
            units (string, ='nm'): 1 BU = 1 nm, by default. Can change
                to angstroms ('A').
            vertices (int, =64): Number of vertices in each bond
                cylinder.
            edge_split (bool, =False): Whether to apply the edge split
                modifier to each bond.

        Returns:
            The new bond (Blender object).
        """

        loc_corr = tuple(c*UNIT_CONV[units] for c in location)
        len_corr = length * UNIT_CONV[units]
        radius_corr = radius * UNIT_CONV[units]

        # Build the cylinder mesh directly with bmesh. Avoids the per-bond
        # depsgraph hit from bpy.ops.mesh.primitive_cylinder_add.
        bm = bmesh.new()
        bmesh.ops.create_cone(bm,
                              cap_ends=False,
                              segments=vertices,
                              radius1=radius_corr,
                              radius2=radius_corr,
                              depth=len_corr)
        for f in bm.faces:
            f.smooth = True

        me = bpy.data.meshes.new("BondMesh")
        bm.to_mesh(me)
        bm.free()

        bond_obj = bpy.data.objects.new("bond_half", me)
        bpy.context.collection.objects.link(bond_obj)
        bpy.context.view_layer.objects.active = bond_obj

        # Position and orient via direct matrix assignment (axis-angle),
        # rather than bpy.ops.transform.rotate on an active selection.
        axis = Vector(tuple(rot_axis / np.linalg.norm(rot_axis)))
        bond_obj.matrix_world = (Matrix.Translation(loc_corr)
                                 @ Matrix.Rotation(rot_angle, 4, axis))

        if edge_split:
            bond_obj.modifiers.new('EdgeSplit', 'EDGE_SPLIT')
            bpy.ops.object.modifier_apply(modifier='EdgeSplit')

        if color == 'by_element':
            bond_color = ELEMENT_COLORS[element]
        else:
            bond_color = color

        if bond_color not in bpy.data.materials:
            _create_new_material(bond_color, COLORS[bond_color])

        bond_obj.data.materials.append(bpy.data.materials[bond_color])

        return bond_obj

    def draw(self, radius=0.2, color='by_element', units='nm',
             vertices=64, edge_split=False):
        """Draw the bond as two half bonds (to allow coloring).

        Args:
            radius (float, =0.2): Radius of cylinder in angstroms.
            color (string, ='by_element'): Color of the bond. If
                'by_element', each half gets element coloring.
            units (string, ='nm'): 1 BU = 1 nm, by default. Can change
                to angstroms ('A').
            vertices (int, =64): Number of vertices in each bond
                cylinder.
            edge_split (bool, =False): Whether to apply the edge split
                modifier to each bond.

        Returns:
            The bond (Blender object), with both halves joined.
        """

        created_objects = []

        center_loc = (self.atom1.location + self.atom2.location)/2
        bond_vector = self.atom1.location - self.atom2.location
        length = np.linalg.norm(bond_vector)

        bond_axis = bond_vector/length
        cyl_axis = np.array((0, 0, 1))
        rot_axis = np.cross(bond_axis, cyl_axis)
        # Fix will not draw bond if perfectly aligned along z axis
        # because rot_axis becomes (0, 0, 0).
        if ((bond_axis == np.array((0, 0, 1))).all()
                or (bond_axis == np.array((0, 0, -1))).all()):
            rot_axis = np.array((1, 0, 0))
        angle = -np.arccos(np.dot(cyl_axis, bond_axis))

        start_center = (self.atom1.location + center_loc)/2
        created_objects.append(Bond._draw_half(start_center, length/2, angle,
                               rot_axis, self.atom1.at_num, radius, color,
                               units, vertices, edge_split))

        end_center = (self.atom2.location + center_loc)/2
        created_objects.append(Bond._draw_half(end_center, length/2, angle,
                               rot_axis, self.atom2.at_num, radius, color,
                               units, vertices, edge_split))

        # Deselect all objects in scene.
        for obj in bpy.context.selected_objects:
            obj.select_set(state=False)
        # Select all newly created objects.
        for obj in created_objects:
            obj.select_set(state=True)

        bpy.ops.object.join()
        bpy.context.object.name = "bond_{}({})_{}({})".format(
            self.atom1.id_num, self.atom1.at_num, self.atom2.id_num,
            self.atom2.at_num)

        return bpy.context.object


class Molecule:
    """The molecule object.

    Attributes:
        atoms (list, = []): List of atoms (atom objects) in molecule.
        bonds (list, = []): List of bonds (bond objects) in molecule.
    """

    def __init__(self, name='molecule', atoms=None, bonds=None):
        self.name = name
        self.atoms = list(atoms) if atoms is not None else []
        self.bonds = list(bonds) if bonds is not None else []
        self._atoms_by_id = {a.id_num: a for a in self.atoms}
        self._bonds_by_key = {
            frozenset((b.atom1.id_num, b.atom2.id_num)): b
            for b in self.bonds
            }

    def add_atom(self, atom):
        """Adds an atom to the molecule."""
        self.atoms.append(atom)
        self._atoms_by_id[atom.id_num] = atom

    def add_bond(self, a1id, a2id):
        """Adds a bond to the molecule, using atom ids."""
        key = frozenset((a1id, a2id))
        if key in self._bonds_by_key:
            return
        bond = Bond(self._atoms_by_id[a1id], self._atoms_by_id[a2id])
        self.bonds.append(bond)
        self._bonds_by_key[key] = bond

    def search_atomid(self, id_to_search):
        """Returns the atom object with the given (unique) id, or None."""
        return self._atoms_by_id.get(id_to_search)

    def search_bondids(self, id1, id2):
        """Returns the bond object between the two (unique) atom ids,
        or None."""
        return self._bonds_by_key.get(frozenset((id1, id2)))

    def draw_bonds(self, caps=True, radius=0.2, color='by_element',
                   units='nm', join=True, with_H=True, subsurf_level=1,
                   vertices=64, edge_split=False):
        """Draws the molecule's bonds.

        Args:
            caps (bool, =True): If true, each bond capped with sphere of
                radius at atom position. Make false if drawing
                ball-and-stick model using separate atom drawings.
            radius (float, =0.2): Radius of bonds in angstroms.
            color (string, ='by_element'): Color of the bonds. If
                'by_element', each gets element coloring.
            units (string, ='nm'): 1 BU = 1 nm, by default. Can change
                to angstroms ('A').
            join (bool, =True): If true, all bonds are joined together
                into a single Bl object.
            with_H (bool, =True): Include H's.
            subsurf_level (int, =1): Subsurface subdivisions that will
            	be applied to the atoms (end caps).
            vertices (int, =64): Number of vertices in each bond
                cylinder.
            edge_split (bool, =False): Whether to apply the edge split
                modifier to each bond.

        Returns:
            The bonds as a single Blender object, if join=True.
            Otherwise, None.
        """

        created_objects = []

        for b in self.bonds:
            if with_H or (b.atom1.at_num != 1 and b.atom2.at_num != 1):
                created_objects.append(b.draw(radius=radius,
                                              color=color,
                                              units=units,
                                              vertices=vertices,
                                              edge_split=edge_split))

        if caps:
            for a in self.atoms:
                if with_H or a.at_num != 1:
                    created_objects.append(a.draw(color=color,
                                                  radius=radius,
                                                  units=units,
                                                  subsurf_level=subsurf_level))

        if join:
            # # Deselect anything currently selected.
            # for obj in bpy.context.selected_objects:
            #     obj.select = False

            # # Select drawn bonds.
            # for obj in created_objects:
            #     obj.select = True
            # Deselect all objects in scene.
            for obj in bpy.context.selected_objects:
                obj.select_set(state=False)
            # Select drawn bonds.
            for obj in created_objects:
                obj.select_set(state=True)

            bpy.ops.object.join()

            bpy.context.object.name = self.name + '_bonds'

            return bpy.context.object

        else:
            return None

    def draw_atoms(self, color='by_element', radius=None, units='nm',
                   scale=1.0, join=True, with_H=True, subsurf_level=2,
                   segments=16):
        """Draw spheres for all atoms.

        Args:
            color (str, ='by_element'): If 'by_element', uses colors in
                ELEMENT_COLORS. Otherwise, can specify color for whole
                model. Must be defined in COLORS.
            radius (float, =None): If specified, gives radius of all
                atoms.
            units (str, ='nm'): Units for 1 BU. Can also be A.
            join (bool, =True): If true, all atoms are joined together
                into a single Bl object.
            with_H (bool, =True): Include the hydrogens.
            subsurf_level (int, =2): Subsurface subdivisions that will
                be applied to the atoms.
            segments (int, =16): Number of segments in each UV sphere
                primitive

        Returns:
            The atoms as a single Blender object, if join=True.
            Otherwise, None.
        """

        # Store start time to time script.
        start_time = time.time()

        # Holds links to all created objects, so that they can be
        # joined.
        created_objects = []

        # Shared per-call mesh cache: one UV sphere per (at_num,
        # resolved-radius-in-BU, segments) tuple, reused as linked
        # instances across atoms.
        mesh_cache = {}

        # Initiate progress monitor over mouse cursor.
        bpy.context.window_manager.progress_begin(0, len(self.atoms))

        n = 0
        for a in self.atoms:
            if with_H or a.at_num != 1:
                created_objects.append(a.draw(color=color, radius=radius,
                                              units=units, scale=scale,
                                              subsurf_level=subsurf_level,
                                              segments=segments,
                                              mesh_cache=mesh_cache))
            n += 1
            bpy.context.window_manager.progress_update(n)

        # End progress monitor.
        bpy.context.window_manager.progress_end()

        joined = None
        if join and created_objects:
            # Deselect all objects in scene.
            for obj in bpy.context.selected_objects:
                obj.select_set(state=False)
            # Select all newly created objects.
            for obj in created_objects:
                obj.select_set(state=True)

            bpy.context.view_layer.objects.active = created_objects[0]
            bpy.ops.object.join()
            bpy.context.object.name = self.name + '_atoms'
            joined = bpy.context.object

        print("{} seconds".format(time.time()-start_time))
        return joined

    def read_pdb(self, filename):
        """Loads a pdb file into a molecule object. Only accepts atoms
        with Cartesian coords through the ATOM/HETATM label and bonds
        through the CONECT label.

        Args:
            filename (string): The target file.
        """

        with open(filename) as pdbfile:
            for line_num, line in enumerate(pdbfile, start=1):
                if line[0:4] == "ATOM" or line[0:6] == "HETATM":
                    symbol = line[76:78].strip().upper()
                    try:
                        atnum = ATOMIC_NUMBERS[symbol]
                    except KeyError:
                        raise ValueError(
                            "Unsupported element {!r} in {} (line {}): {}"
                            .format(symbol, filename, line_num,
                                    line.rstrip()))
                    idnum = int(line[6:11])
                    coords = np.array((float(line[30:38]), float(line[38:46]),
                                       float(line[46:54])))
                    self.add_atom(Atom(atnum, coords, idnum))

                elif line[0:6] == "CONECT":

                    # Loads atoms as a list. First atom is bonded to the
                    # remaining atoms (up to four).
                    atoms = line[6:].split()
                    for bonded_atom in atoms[1:]:
                        # print(atoms[0], bonded_atom)
                        self.add_bond(int(atoms[0]), int(bonded_atom))
