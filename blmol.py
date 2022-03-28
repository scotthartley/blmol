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
import time, mathutils

# Python outside of Blender doesn't play all that well with bpy, so need
# to handle ImportError.
try:
    import bpy
    import bmesh
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

class TemplateCollection:

    """A collection for templates in order to copy them when needed. All the
    templates are stored in self.coll with structure {at_num:Blender object}.

    Get copy with self.get_copy(at_num).
    Delete collection with self.delete().
    """

    def __init__(self):
        self.coll = None
    def get_copy(self, at_num, copy_data=True):
        """Copies and returns the template corresponding to at_num

        Args:
            copy_data (bool, =True): Whether the object's data, i. e. its mesh
                is copied or the identical mesh instance is used."""
        copy = self.coll[at_num].copy()
        if copy_data:
            copy.data = copy.data.copy()
        return copy
    def delete(self):
        """Deletes all objects stored in self.coll"""
        # Deselect all
        bpy.ops.object.select_all(action='DESELECT')
        # Select all template objects
        for obj in self.coll.values():
            bpy.data.meshes.remove(obj.data)
        #     obj.select_set(True)
        # bpy.ops.object.delete()
        self.coll = None

class AtomTemplateCollection(TemplateCollection):

    __doc__ = TemplateCollection.__doc__ + """
    color (str, ='by_element'): If 'by_element', uses colors in
        ELEMENT_COLORS. Otherwise, can specify color for whole
        model. Must be defined in COLORS.
    radius (float, =None): If specified, gives radius of all
        atoms.
    units (str, ='nm'): Units for 1 BU. Can also be A.
    scale (float, =1.0):
    subsurf_level (int, =2): Subsurface subdivisions that will
        be applied to the atoms.
    segments (int, =16): Number of segments in each UV sphere
        primitive
    template_collection (AtomTemplateCollection, =None):
        If specified, the spheres will be copies from the given
        template collection and all the other arguments will be
        ignored.
    """

    def __init__(self, color='by_element', radius=None, units='nm',
                 scale=1.0, subsurf_level=2, segments=16):
        super(AtomTemplateCollection, self).__init__()
        self.color = color
        self.radius = radius
        self.units = units
        self.scale = scale
        self.subsurf_level = subsurf_level
        self.segments = segments

        self.create_template_collection()
    def create_template_collection(self):
        scale_factor = UNIT_CONV[self.units]*self.scale
        self.coll = {}
        for at_num in ATOMIC_NUMBERS.values():
            # Work out the sphere radius in BU.
            if not self.radius:
                rad_adj = RADII[at_num]*scale_factor
            else:
                rad_adj = self.radius*scale_factor

            # Create sphere as bmesh.
            bm = bmesh.new()
            bmesh.ops.create_uvsphere(bm,
                                      u_segments=self.segments,
                                      v_segments=self.segments,
                                      radius=rad_adj)

            for f in bm.faces:
                f.smooth = True

            # Convert to mesh.
            me = bpy.data.meshes.new("Mesh")
            bm.to_mesh(me)
            bm.free()

            # Assign mesh to object and place in space.
            atom_sphere = bpy.data.objects.new("template({})_atom({})".format(
                                id(self), at_num), me)
            bpy.context.collection.objects.link(atom_sphere)

            # Assign subsurface modifier, if requested
            if self.subsurf_level != 0:
                atom_sphere.modifiers.new('Subsurf', 'SUBSURF')
                atom_sphere.modifiers['Subsurf'].levels = self.subsurf_level

            # Color atom and assign material
            if self.color == 'by_element':
                atom_color = ELEMENT_COLORS[at_num]
            else:
                atom_color = self.color

            if atom_color not in bpy.data.materials:
                _create_new_material(atom_color, COLORS[atom_color])

            atom_sphere.data.materials.append(bpy.data.materials[atom_color])

            # Add new object to collection
            self.coll[at_num] = atom_sphere

class BondTemplateCollection(TemplateCollection):

    __doc__ = TemplateCollection.__doc__ + """
    radius (float, =0.2): Radius of bonds in angstroms.
    color (string, ='by_element'): Color of the bonds. If
        'by_element', each gets element coloring.
    units (string, ='nm'): 1 BU = 1 nm, by default. Can change
        to angstroms ('A').
    vertices (int, =64): Number of vertices in each bond
        cylinder.
    edge_split (bool, =False): Whether to apply the edge split
        modifier to each bond.
    """

    def __init__(self, radius=0.2, color='by_element', units='nm',
                 vertices=64, edge_split=False):
        super(BondTemplateCollection, self).__init__()
        self.radius = radius
        self.color = color
        self.units = units
        self.vertices = vertices
        self.edge_split = edge_split

        self.create_template_collection()
    def create_template_collection(self):
        self.coll = {}
        for at_num in ATOMIC_NUMBERS.values():
            radius_corr = self.radius * UNIT_CONV[self.units]

            bpy.ops.mesh.primitive_cylinder_add(vertices=self.vertices,
                                                radius=radius_corr,
                                                depth=1,
                                                end_fill_type='NOTHING')

            bpy.ops.object.shade_smooth()

            if self.edge_split:
                bpy.ops.object.modifier_add(type='EDGE_SPLIT')
                bpy.ops.object.modifier_apply(modifier='EdgeSplit')

            if self.color == 'by_element':
                bond_color = ELEMENT_COLORS[at_num]
            else:
                bond_color = self.color

            if bond_color not in bpy.data.materials:
                _create_new_material(bond_color, COLORS[bond_color])

            bpy.context.object.data.materials.append(
                bpy.data.materials[bond_color])

            bond_cylinder = bpy.context.active_object
            bond_cylinder.name = "template({})_bond({})".format(
                                id(self), at_num)
            self.coll[at_num] = bond_cylinder

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

    def draw(self, template_collection, copy_data=True):
        """Draw the atom in Blender.

        Args:
            template_collection (AtomTemplateCollection)

        Returns:
            The blender object.
        """

        # The corrected location (i.e., scaled to units.)
        scale_factor = UNIT_CONV[template_collection.units]
        loc_corr = tuple(c*scale_factor for c in self.location)

        atom_sphere_copy = template_collection.get_copy(self.at_num, copy_data)

        atom_sphere_copy.location = loc_corr
        atom_sphere_copy.name = "atom({})_{}".format(self.at_num, self.id_num)

        return atom_sphere_copy


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
                   template_collection, copy_data=True):
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
            template_collection (BondTemplateCollection)

        Returns:
            The new bond (Blender object).
        """

        scale_factor = UNIT_CONV[template_collection.units]
        loc_corr = tuple(c*scale_factor for c in location)
        len_corr = length * scale_factor

        bond_copy = template_collection.get_copy(element, copy_data)

        bond_copy.location = loc_corr

        # Resize bond
        bond_copy.scale = (1, 1, len_corr)

        mat_rot = mathutils.Matrix.Rotation(rot_angle, 3, rot_axis)
        bond_copy.rotation_euler = tuple(mat_rot.to_euler())

        return bond_copy

    def draw(self, template_coll, join_halves=False, copy_data=True):
        """Draw the bond as two half bonds (to allow coloring).

        Args:
            template_coll (BondTemplateCollection)
            join_halves (bool, =False): Join both halves to one object. Setting
                it to True will slow down the code as bpy.ops methods are used.

        Returns:
            List of the two unjoined halves (Blender objects) when
            join_halves=False. Otherwise one-tuple with joined halves.
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

        # Create both halves
        for a1, a2 in ((self.atom1, self.atom2), (self.atom2, self.atom1)):
            center_of_half = (a1.location + center_loc)/2
            created_objects.append(Bond._draw_half(center_of_half, length/2,
                                   angle, rot_axis, a1.at_num, template_coll,
                                   copy_data))
            if not join_halves:
                created_objects[-1].name = "bond_{}({})_{}({})".format(
                    a1.id_num, a1.at_num, a2.id_num, a2.at_num)

        if join_halves:
            # Deselect all objects in scene.
            for obj in bpy.context.selected_objects:
                obj.select_set(state=False)
            # Select all newly created objects.
            for obj in created_objects:
                bpy.context.collection.objects.link(obj)
                obj.select_set(state=True)

            obj = created_objects[0]
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.join()
            obj.name = "bond_{}({})_{}({})".format(self.atom1.id_num,
                self.atom1.at_num, self.atom2.id_num, self.atom2.at_num)

            # An iterable is expected thus return one-tuple
            return (obj,)

        return created_objects


class Molecule:
    """The molecule object.

    Attributes:
        atoms (list, = []): List of atoms (atom objects) in molecule.
        bonds (list, = []): List of bonds (bond objects) in molecule.
    """

    def __init__(self, name='molecule', atoms=None, bonds=None):
        self.name = name
        if atoms is None:
            self.atoms = []
        else:
            self.atoms = atoms
        if bonds is None:
            self.bonds = []
        else:
            self.bonds = bonds

    def add_atom(self, atom):
        """Adds an atom to the molecule."""
        self.atoms.append(atom)

    def add_bond(self, a1id, a2id):
        """Adds a bond to the molecule, using atom ids."""
        if not self.search_bondids(a1id, a2id):
            self.bonds.append(Bond(self.search_atomid(a1id),
                                   self.search_atomid(a2id)))

    def search_atomid(self, id_to_search):
        """Searches through atom list and returns atom object
        corresponding to (unique) id."""
        for atom in self.atoms:
            if atom.id_num == id_to_search:
                return atom
        return None

    def search_bondids(self, id1, id2):
        """Searches through bond list and returns bond object
        corresponding to (unique) ids."""
        for b in self.bonds:
            if ((id1, id2) == (b.atom1.id_num, b.atom2.id_num) or
                    (id2, id1) == (b.atom1.id_num, b.atom2.id_num)):
                return b
        return None

    def draw_bonds(self, caps=True, radius=0.2, color='by_element',
                   units='nm', join=True, join_halves=False, with_H=True,
                   segments=16, subsurf_level=1, vertices=64, edge_split=False,
                   bond_template_coll=None, atom_template_coll=None):
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
            join_halves (bool, =False): Join both halves to one object. Setting
                it to True will slow down the code as bpy.ops methods are used.
                Has no effect when join=True.
            with_H (bool, =True): Include H's.
            segments (int, =16)
            subsurf_level (int, =1): Subsurface subdivisions that will
            	be applied to the atoms (end caps).
            vertices (int, =64): Number of vertices in each bond
                cylinder.
            edge_split (bool, =False): Whether to apply the edge split
                modifier to each bond.
            bond_template_coll (BondTemplateCollection, =None):
                If specified, the bonds will be copies from the given
                template collection and all the arguments regarding the bond's
                mesh will be ignored.
            atom_template_coll (AtomTemplateCollection, =None):
                If specified, the atoms will be copies from the given
                template collection and all the arguments regarding the atom's
                mesh will be ignored.

        Returns:
            The bonds as a single Blender object, if join=True.
            Otherwise, None.
        """

        # Store start time to time script.
        start_time = time.time()

        collection = bpy.context.collection

        # Create AtomTemplateCollection if necessary
        newly_created_template_colls = []
        if not atom_template_coll:
            atom_template_coll = AtomTemplateCollection(color=color,
                radius=radius, units=units, subsurf_level=subsurf_level,
                segments=segments)
            newly_created_template_colls.append(atom_template_coll)
        if not bond_template_coll:
            bond_template_coll = BondTemplateCollection(radius=radius,
                color=color, units=units, vertices=vertices,
                edge_split=edge_split)
            newly_created_template_colls.append(bond_template_coll)

        join_halves = join_halves and not join
        copy_data = not join

        created_objects = []

        for b in self.bonds:
            if with_H or (b.atom1.at_num != 1 and b.atom2.at_num != 1):
                new_halves = b.draw(bond_template_coll, join_halves, copy_data)

                # Add new objects to internal list
                created_objects += new_halves

                # Link new objects with collection
                if not join_halves:
                    for new_half in new_halves:
                        collection.objects.link(new_half)

        if caps:
            for a in self.atoms:
                if with_H or a.at_num != 1:
                    new_atom = a.draw(atom_template_coll, copy_data)

                    # Add new objects to internal list
                    created_objects.append(new_atom)

                    # Link new objects with collection
                    collection.objects.link(new_atom)

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

            bpy.context.view_layer.objects.active = created_objects[0]

            # Copy data of active object so the templates are unaffected of join
            bpy.context.object.data = bpy.context.object.data.copy()

            bpy.ops.object.join()
            bpy.context.object.name = self.name + '_bonds'
            bpy.context.object.data.name = self.name + '_bonds'

            ret = bpy.context.object

        else:
            ret = None

        # Clean up in case a new template collection was created
        for new_template_coll in newly_created_template_colls:
            new_template_coll.delete()

        print("{} seconds".format(time.time()-start_time))
        return ret

    def draw_atoms(self, color='by_element', radius=None, units='nm',
                   scale=1.0, join=True, with_H=True, subsurf_level=2,
                   segments=16, template_collection=None):
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
            template_collection (AtomTemplateCollection, =None):
                If specified, the spheres will be copies from the given
                template collection and all the other arguments will be
                ignored.

        Returns:
            The atoms as a single Blender object, if join=True.
            Otherwise, None.
        """

        # Store start time to time script.
        start_time = time.time()

        collection = bpy.context.collection

        # Create AtomTemplateCollection if necessary
        if not template_collection:
            created_new_template_collection = True
            template_collection = AtomTemplateCollection(color=color,
                radius=radius, units=units, scale=scale,
                subsurf_level=subsurf_level, segments=segments)
        else:
            created_new_template_collection = False

        copy_data = not join

        # Holds links to all created objects, so that they can be
        # joined.
        created_objects = []

        # Initiate progress monitor over mouse cursor.
        bpy.context.window_manager.progress_begin(0, len(self.atoms))

        n = 0
        for a in self.atoms:
            if with_H or a.at_num != 1:
                atom_sphere = a.draw(template_collection, copy_data)

                # Add new objects to internal list
                created_objects.append(atom_sphere)

                # Link new objects with collection
                collection.objects.link(atom_sphere)

            n += 1
            bpy.context.window_manager.progress_update(n)

        # End progress monitor.
        bpy.context.window_manager.progress_end()

        if join:
            # Deselect all objects in scene.
            for obj in bpy.context.selected_objects:
                obj.select_set(state=False)
            # Select all newly created objects.
            for obj in created_objects:
                obj.select_set(state=True)

            bpy.context.view_layer.objects.active = created_objects[0]

            # Copy data of active object so the templates are unaffected of join
            bpy.context.object.data = bpy.context.object.data.copy()

            bpy.ops.object.join()
            bpy.context.object.name = self.name + '_atoms'
            bpy.context.object.data.name = self.name + '_atoms'

            ret = bpy.context.object

        else:
            ret = None

        # Clean up in case a new template collection was created
        if created_new_template_collection:
            template_collection.delete()

        print("{} seconds".format(time.time()-start_time))
        return ret

    def read_pdb(self, filename):
        """Loads a pdb file into a molecule object. Only accepts atoms
        with Cartesian coords through the ATOM/HETATM label and bonds
        through the CONECT label.

        Args:
            filename (string): The target file.
        """

        with open(filename) as pdbfile:
            for line in pdbfile:
                if line[0:4] == "ATOM":
                    idnum = int(line[6:11])
                    atnum = ATOMIC_NUMBERS[line[76:78].strip().upper()]
                    coords = np.array((float(line[30:38]), float(line[38:46]),
                                       float(line[46:54])))
                    self.add_atom(Atom(atnum, coords, idnum))

                elif line[0:6] == "HETATM":
                    idnum = int(line[6:11])
                    atnum = ATOMIC_NUMBERS[line[76:78].strip().upper()]
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
