#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import lib.nbo_field as nbo
import linecache

class MolecularFieldAnalysis:
    def __init__(self, mean_plane = [0,1,2,3,4,5], origin = 2, x_axis = 5, n = [0,1,2,3,4,6], grid_x=6, grid_y=8, grid_z=8, threshold=0.01):
        self.mean_plane = mean_plane
        self.origin = origin
        self.x_axis = x_axis
        self.n = n
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        self.threshold = threshold
        self.unitcell_size = 1

    def indicator_field(self, mol_name):
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        Y = self._standard_xyz(mol_name)
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        N_samples = len(mol_name_train)
        for i in range(0,N_samples):
            Atoms = np.loadtxt(mol_name_train[i],skiprows=2,usecols=(0,),unpack=True,dtype='str')
            Z = self._align_xyz(mol_name, i, Y)
            N_atoms = self._atom_number(mol_name, i)
            indicator_field = mol_field.calc_field(Z,Atoms,N_atoms)
            if i == 0:
                indicator_fields = indicator_field
            else:
                indicator_fields = np.c_[indicator_fields,indicator_field]
        indicator_fields = indicator_fields.T
        descriptor = mol_field.prescreening(N_samples,indicator_fields)
        return (indicator_fields, descriptor)

    def indicator_field_sub(self, mol_name, Y):
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        N_samples = len(mol_name_train)
        for i in range(0,N_samples):
            Atoms = np.loadtxt(mol_name_train[i],skiprows=2,usecols=(0,),unpack=True,dtype='str')
            Z = self._align_xyz(mol_name, i, Y)
            N_atoms = self._atom_number(mol_name, i)
            indicator_field = mol_field.calc_field(Z,Atoms,N_atoms)
            if i == 0:
                indicator_fields = indicator_field
            else:
                indicator_fields = np.c_[indicator_fields,indicator_field]
        indicator_fields = indicator_fields.T
        descriptor = mol_field.prescreening(N_samples,indicator_fields)
        return (indicator_fields, descriptor)

    def indicator_field_test(self, mol_name,mol_name_pred):
        indicator_fields = self.indicator_field(mol_name)
        indicator_fields = indicator_fields[0]
        mol_name_test = np.loadtxt(mol_name_pred,dtype='str')
        Y = self._standard_xyz(mol_name)
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        N_samples = len(mol_name_test)
        for i in range(0,N_samples):
            mol_name = mol_name_test[i]
            Atoms = np.loadtxt(mol_name,skiprows=2,usecols=(0,),unpack=True,dtype='str')
            Z = self._align_xyz(mol_name_test, i, Y)
            N_atoms = self._atom_number(mol_name_test, i)
            indicator_field_test = mol_field.calc_field(Z,Atoms,N_atoms)
            if i == 0:
                indicator_fields_test = indicator_field_test
            else:
                indicator_fields_test = np.c_[indicator_fields_test,indicator_field_test]
        indicator_fields_test = indicator_fields_test.T
        descriptor = mol_field.prescreening_test(N_samples,indicator_fields,indicator_fields_test)
        return (indicator_fields_test, descriptor)

    def indicator_field_corr(self, mol_name,target_variable):
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        Y = self._standard_xyz(mol_name)
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        N_samples = len(mol_name_train)
        for i in range(0,N_samples):
            Atoms = np.loadtxt(mol_name_train[i],skiprows=2,usecols=(0,),unpack=True,dtype='str')
            Z = self._align_xyz(mol_name, i, Y)
            N_atoms = self._atom_number(mol_name, i)
            indicator_field = mol_field.calc_field(Z,Atoms,N_atoms)
            if i == 0:
                indicator_fields = indicator_field
            else:
                indicator_fields = np.c_[indicator_fields,indicator_field]
        indicator_fields = indicator_fields.T
        descriptor_pre = mol_field.prescreening(N_samples,indicator_fields)
        j = 0
        n = len(descriptor_pre[:,0])-1
        for i in range(len(descriptor_pre[0,:])):
            corr = (np.corrcoef(descriptor_pre[1:n+1,i],target_variable))
            if np.abs(corr[1,0]) > 0.3:
                if j == 0:
                    descriptor = descriptor_pre[:,i]
                    j += 1
                else:
                    descriptor = np.c_[descriptor,descriptor_pre[:,i]]
        return (indicator_fields, descriptor)

    def indicator_field_test_sub(self, mol_name,mol_name_pred, Y):
        indicator_fields = self.indicator_field_sub(mol_name,Y)
        indicator_fields = indicator_fields[0]
        mol_name_test = np.loadtxt(mol_name_pred,dtype='str')
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        N_samples = len(mol_name_test)
        for i in range(0,N_samples):
            mol_name = mol_name_test[i]
            Atoms = np.loadtxt(mol_name,skiprows=2,usecols=(0,),unpack=True,dtype='str')
            Z = self._align_xyz(mol_name_test, i, Y)
            N_atoms = self._atom_number(mol_name_test, i)
            indicator_field_test = mol_field.calc_field(Z,Atoms,N_atoms)
            if i == 0:
                indicator_fields_test = indicator_field_test
            else:
                indicator_fields_test = np.c_[indicator_fields_test,indicator_field_test]
        indicator_fields_test = indicator_fields_test.T
        descriptor = mol_field.prescreening_test(N_samples,indicator_fields,indicator_fields_test)
        return (indicator_fields_test, descriptor)

    def indicator_field_corr_test(self, mol_name,target_variable,sample_number):
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        Y = self._standard_xyz(mol_name)
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        N_samples = len(mol_name_train)
        for i in range(0,N_samples):
            Atoms = np.loadtxt(mol_name_train[i],skiprows=2,usecols=(0,),unpack=True,dtype='str')
            Z = self._align_xyz(mol_name, i, Y)
            N_atoms = self._atom_number(mol_name, i)
            indicator_field = mol_field.calc_field(Z,Atoms,N_atoms)
            if i == 0:
                indicator_fields = indicator_field
            else:
                indicator_fields = np.c_[indicator_fields,indicator_field]
        indicator_fields = indicator_fields.T
        j = 0
        for i in range(len(indicator_fields[0,:])):
            corr = (np.corrcoef(indicator_fields[0:sample_number,i],target_variable))
            if np.abs(corr[1,0]) > 0.3:
                if j == 0:
                    descriptor = indicator_fields[:,i]
                    j += 1
                else:
                    descriptor = np.c_[descriptor,indicator_fields[:,i]]
        return (indicator_fields, descriptor)

    def structure_info(self, coefficient,mol_name,sample_number,indicator_fields):
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        coordinate = mol_field.unitcell_coordinate()
        coef = np.loadtxt(coefficient, skiprows=1, usecols=(1,),unpack=True)
        Vis = mol_field.visualization_preprocess(coordinate, indicator_fields, coef)
        important_coordinate = mol_field.structual_information_coordinate(Vis, self.threshold)
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        Y = self._standard_xyz(mol_name)
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        Atoms = np.loadtxt(mol_name_train[sample_number],skiprows=2,usecols=(0,),unpack=True,dtype='str')
        Z = self._align_xyz(mol_name, sample_number, Y)
        N_atoms = self._atom_number(mol_name, sample_number)
        indicator_field = mol_field.calc_field(Z,Atoms,N_atoms)
        indicator_fields = indicator_fields.T
        important_information = mol_field.structual_information(indicator_field, Vis, self.threshold)
        return (important_coordinate,important_information)

    def structure_info_sub(self, coefficient,mol_name,sample_number,indicator_fields, Y):
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        coordinate = mol_field.unitcell_coordinate()
        coef = np.loadtxt(coefficient, skiprows=1, usecols=(1,),unpack=True)
        Vis = mol_field.visualization_preprocess(coordinate, indicator_fields, coef)
        important_coordinate = mol_field.structual_information_coordinate(Vis, self.threshold)
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        Atoms = np.loadtxt(mol_name_train[sample_number],skiprows=2,usecols=(0,),unpack=True,dtype='str')
        Z = self._align_xyz(mol_name, sample_number, Y)
        N_atoms = self._atom_number(mol_name, sample_number)
        indicator_field = mol_field.calc_field(Z,Atoms,N_atoms)
        indicator_fields = indicator_fields.T
        important_information = mol_field.structual_information(indicator_field, Vis, self.threshold)
        return (important_coordinate,important_information)

    def structure_info_sub_corr(self, coefficient,mol_name,sample_number,indicator_fields, Y, target_variable,corr_threshold):
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        coordinate = mol_field.unitcell_coordinate()
        coef = np.loadtxt(coefficient, skiprows=1, usecols=(1,),unpack=True)
        Vis = mol_field.visualization_preprocess(coordinate, indicator_fields, coef)
        for i in range(len(indicator_fields[0,:])):
            corr = (np.corrcoef(indicator_fields[range(len(target_variable)),i],target_variable))
            corr = np.nan_to_num(corr,nan = 0)
            if abs(corr[1,0])<=corr_threshold or np.sign(corr[1,0]) != np.sign(Vis[i,0]):
                Vis[i,0] = 0
        important_coordinate = mol_field.structual_information_coordinate(Vis, self.threshold)
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        Atoms = np.loadtxt(mol_name_train[sample_number],skiprows=2,usecols=(0,),unpack=True,dtype='str')
        Z = self._align_xyz(mol_name, sample_number, Y)
        N_atoms = self._atom_number(mol_name, sample_number)
        indicator_field = mol_field.calc_field(Z,Atoms,N_atoms)
        indicator_fields = indicator_fields.T
        important_information = mol_field.structual_information(indicator_field, Vis, self.threshold)
        return (important_coordinate,important_information)

    def structure_info_corr(self, coefficient,mol_name,sample_number,indicator_fields,target_variable):
        N_samples = len(target_variable)
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        coordinate = mol_field.unitcell_coordinate()
        coef = np.loadtxt(coefficient, skiprows=1, usecols=(1,),unpack=True)
        Vis = mol_field.visualization_preprocess_corr(coordinate, indicator_fields, coef, target_variable,N_samples)
        important_coordinate = mol_field.structual_information_coordinate(Vis, self.threshold)
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        Y = self._standard_xyz(mol_name)
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        Atoms = np.loadtxt(mol_name_train[sample_number],skiprows=2,usecols=(0,),unpack=True,dtype='str')
        Z = self._align_xyz(mol_name, sample_number, Y)
        N_atoms = self._atom_number(mol_name, sample_number)
        indicator_field = mol_field.calc_field(Z,Atoms,N_atoms)
        indicator_fields = indicator_fields.T
        important_information = mol_field.structual_information(indicator_field, Vis, self.threshold)
        return (important_coordinate,important_information)

    def xyz_file(self,mol_name,sample_number,important_information,Directory):
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        Y = self._standard_xyz(mol_name)
        Z = self._align_xyz(mol_name, sample_number, Y)
        N_atoms = self._atom_number(mol_name, sample_number)
        mol_name = mol_name_train[sample_number]
        mv_name = Directory + mol_name.replace("data/","")
        atoms = np.loadtxt(mol_name,skiprows=2,usecols=(0,),unpack=True,dtype='S3')
        coordinate = important_information[0]
        label = important_information[1]
        n_all = N_atoms + len(important_information[1])
        with open(mv_name,'w') as f:
            f.write(str(n_all)+'\n'+mv_name+'\n')
            for i in range(N_atoms):
                f.write(str((atoms.T[i]).decode())+'\t'+str(Z[i,0])+'\t'+str(Z[i,1])+'\t'+str(Z[i,2])+'\n')
            for i in range(len(important_information[1])):
                f.write(str(label[i])+'\t'+str(coordinate[i,0])+'\t'+str(coordinate[i,1])+'\t'+str(coordinate[i,2])+'\n')

    def xyz_file_sub(self,mol_name,sample_number,important_information,Directory, Y):
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        Z = self._align_xyz(mol_name, sample_number, Y)
        N_atoms = self._atom_number(mol_name, sample_number)
        mol_name = mol_name_train[sample_number]
        mv_name = Directory + mol_name.replace("data/","")
        atoms = np.loadtxt(mol_name,skiprows=2,usecols=(0,),unpack=True,dtype='S3')
        coordinate = important_information[0]
        label = important_information[1]
        n_all = N_atoms + len(important_information[1])
        with open(mv_name,'w') as f:
            f.write(str(n_all)+'\n'+mv_name+'\n')
            for i in range(N_atoms):
                f.write(str((atoms.T[i]).decode())+'\t'+str(Z[i,0])+'\t'+str(Z[i,1])+'\t'+str(Z[i,2])+'\n')
            for i in range(len(important_information[1])):
                f.write(str(label[i])+'\t'+str(coordinate[i,0])+'\t'+str(coordinate[i,1])+'\t'+str(coordinate[i,2])+'\n')

    def prediction(self, coefficient,mol_name,sample_number,indicator_fields):
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        coordinate = mol_field.unitcell_coordinate()
        intercept =np.loadtxt(coefficient, usecols=(1,),unpack=True)
        intercept = intercept[0]
        coef = np.loadtxt(coefficient, skiprows=1, usecols=(1,),unpack=True)
        coef = mol_field.visualization_preprocess(coordinate, indicator_fields, coef)
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        Y = self._standard_xyz(mol_name)
        mol_field = mfa.indicator_field(self.grid_x,self.grid_y,self.grid_z,self.unitcell_size)
        Atoms = np.loadtxt(mol_name_train[sample_number],skiprows=2,usecols=(0,),unpack=True,dtype='str')
        Z = self._align_xyz(mol_name, sample_number, Y)
        N_atoms = self._atom_number(mol_name, sample_number)
        indicator_field = mol_field.calc_field(Z,Atoms,N_atoms)
        pred_value = mol_field.prediction(indicator_field, coef) + intercept
        return (pred_value)

    def R2q2(self,outcomes):
        measured,pred,pred_LOOCV = np.loadtxt(outcomes+"_output.csv",skiprows=1, usecols=(1,2,3,),unpack=True,delimiter=",")
        R2 = 1-sum((measured-pred)**2)/sum((measured-np.average(measured))**2)
        q2 = 1-sum((measured-pred_LOOCV)**2)/sum((measured-np.average(measured))**2)
        print (outcomes,"R2:",round(R2,3),"q2:",round(q2,3))

    def _standard_xyz(self,mol_name):
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        X = np.loadtxt(mol_name_train[0],skiprows=2,usecols=(1,2,3))
        N_atoms = linecache.getline(mol_name_train[0], int(1))
        N_atoms = int(N_atoms)
        mol = mfa.alignment(N_atoms,self.mean_plane,self.origin,self.x_axis,self.n)
        Y = mol.def_plane(X)
        return Y

    def _align_xyz(self,mol_name,sample_number,Y):
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        N_atoms = linecache.getline(mol_name_train[0], int(1))
        N_atoms = int(N_atoms)
        mol = mfa.alignment(N_atoms,self.mean_plane,self.origin,self.x_axis,self.n)
        mol_name = mol_name_train[sample_number]
        Z = np.loadtxt(mol_name,skiprows=2,usecols=(1,2,3))
        N_atoms = linecache.getline(mol_name, int(1))
        N_atoms = int(N_atoms)
        Z = mol.align_mol(Y, Z, N_atoms)
        return Z


    def _atom_number(self,mol_name,sample_number):
        mol_name_train = np.loadtxt(mol_name,dtype='str')
        mol_name = mol_name_train[sample_number]
        N_atoms = linecache.getline(mol_name, int(1))
        N_atoms = int(N_atoms)
        return N_atoms

# ============================================================
# NBO-field I/O utilities
# ============================================================
def write_overlay_xyz(
    filename,
    elements,
    coordinates,
    dummy_elements,
    voxel_xyz,
    comment="NBO-MFA voxel overlay"
):
    """
    Write an XYZ file containing the original molecular structure followed by
    visualization voxels represented by dummy elements (He/Ne/Ar/Kr).
    """
    n_atoms = len(elements)
    n_voxels = len(dummy_elements)

    if len(coordinates) != n_atoms:
        raise ValueError(
            "Number of elements and atomic coordinates do not match."
        )

    if len(voxel_xyz) != n_voxels:
        raise ValueError(
            "Number of dummy elements and voxel coordinates do not match."
        )

    total = n_atoms + n_voxels

    with open(filename, "w") as f:
        f.write(f"{total}\n")
        f.write(f"{comment}\n")

        for elem, coord in zip(elements, coordinates):
            f.write(
                f"{elem} "
                f"{coord[0]:.6f} "
                f"{coord[1]:.6f} "
                f"{coord[2]:.6f}\n"
            )

        for elem, coord in zip(dummy_elements, voxel_xyz):
            f.write(
                f"{elem} "
                f"{coord[0]:.6f} "
                f"{coord[1]:.6f} "
                f"{coord[2]:.6f}\n"
            )


def write_overlay_tsv(
    filename,
    categories,
    voxel_xyz,
    selected_coef,
    selected_q,
    selected_owner,
    elements
):
    """
    Write detailed information for visualization voxels.

    owner indices returned from C++ are 0-origin internally.
    owner_atom_index in the TSV is written as 1-origin to match XYZ atom
    numbering.
    """
    n = len(categories)

    if not (
        len(voxel_xyz) == n
        and len(selected_coef) == n
        and len(selected_q) == n
        and len(selected_owner) == n
    ):
        raise ValueError(
            "Visualization arrays have inconsistent lengths."
        )

    with open(filename, "w") as f:
        f.write(
            "cat\tx\ty\tz\tcoef\tq_assigned\t"
            "owner_atom_index\towner_element\n"
        )

        for cat, coord, c, q, owner_index in zip(
            categories,
            voxel_xyz,
            selected_coef,
            selected_q,
            selected_owner
        ):
            owner_index = int(owner_index)

            if owner_index >= 0:
                owner_element = elements[owner_index]
                owner_label = owner_index + 1
            else:
                owner_element = "NA"
                owner_label = 0

            f.write(
                f"{cat}\t"
                f"{coord[0]:.6f}\t"
                f"{coord[1]:.6f}\t"
                f"{coord[2]:.6f}\t"
                f"{c:.12g}\t"
                f"{q:.12g}\t"
                f"{owner_label}\t"
                f"{owner_element}\n"
            )


# ============================================================
# NBO-field analysis
# ============================================================
class NBOFieldAnalysis:
    """
    Python wrapper for lib.nbo_field.

    Numerical field construction and voxel classification are performed in
    C++; file I/O and batch execution are handled here in Python.
    """

    def __init__(
        self,
        grid_x=8,
        grid_y=8,
        grid_z=8,
        threshold=0.01,
        unitcell_size=1.0,
        require_nonzero_q=True,
        eps_q=1e-12
    ):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        self.threshold = threshold
        self.unitcell_size = unitcell_size
        self.require_nonzero_q = require_nonzero_q
        self.eps_q = eps_q

    def _new_field(self):
        return nbo.nbo_field(
            self.grid_x,
            self.grid_y,
            self.grid_z,
            self.unitcell_size
        )

    def _mol_list(self, mol_name):
        mol_list = np.loadtxt(mol_name, dtype=str)
        return np.atleast_1d(mol_list)

    def _read_xyz_nbo(self, filename):
        """
        Read a 2-line-header XYZ file with NBO charge in column 5:
            Elem x y z q
        """
        elements = np.loadtxt(
            filename,
            skiprows=2,
            usecols=(0,),
            dtype=str
        )
        coordinates = np.loadtxt(
            filename,
            skiprows=2,
            usecols=(1, 2, 3),
            dtype=float
        )
        charges = np.loadtxt(
            filename,
            skiprows=2,
            usecols=(4,),
            dtype=float
        )

        elements = np.atleast_1d(elements)
        coordinates = np.atleast_2d(coordinates)
        charges = np.atleast_1d(charges)

        if not (
            len(elements) == len(coordinates) == len(charges)
        ):
            raise ValueError(
                f"Inconsistent atom counts in NBO XYZ file: {filename}"
            )

        return elements, coordinates, charges

    def _n_voxel(self):
        nx = int(round(self.grid_x / self.unitcell_size))
        ny = int(round(self.grid_y / self.unitcell_size))
        nz = int(round(self.grid_z / self.unitcell_size))

        if not np.isclose(nx * self.unitcell_size, self.grid_x):
            raise ValueError("grid_x must be divisible by unitcell_size.")
        if not np.isclose(ny * self.unitcell_size, self.grid_y):
            raise ValueError("grid_y must be divisible by unitcell_size.")
        if not np.isclose(nz * self.unitcell_size, self.grid_z):
            raise ValueError("grid_z must be divisible by unitcell_size.")

        return nx * ny * nz

    def _read_coefficient(self, filename):
        """
        Read:
            (Intercept) value
            V1 value
            ...
            Vn value

        V1 corresponds to flattened voxel index 0.
        """
        n_voxel = self._n_voxel()

        intercept = 0.0
        coefficients = np.zeros(n_voxel, dtype=float)
        seen_variables = set()
        max_variable_id = 0

        with open(filename, "r") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) < 2:
                    continue

                key = parts[0]
                value = float(parts[1])

                if key.startswith("(Intercept"):
                    intercept = value

                elif key.startswith("V"):
                    try:
                        variable_id = int(key[1:])
                    except ValueError:
                        continue

                    index = variable_id - 1
                    max_variable_id = max(
                        max_variable_id,
                        variable_id
                    )
                    seen_variables.add(variable_id)

                    if 0 <= index < n_voxel:
                        coefficients[index] = value

        if (
            max_variable_id != n_voxel
            or len(seen_variables) != n_voxel
        ):
            raise ValueError(
                "Coefficient file and grid do not match: "
                f"V1..V{max_variable_id} "
                f"({len(seen_variables)} variables), "
                f"grid={n_voxel} voxels."
            )

        return intercept, coefficients

    def descriptor(
        self,
        mol_name,
        output_file=None,
        fmt="%.10f"
    ):
        """
        Calculate NBO-field descriptors for every XYZ listed in mol_name.

        Returns
        -------
        descriptor : ndarray, shape (n_samples, n_voxels)
        """
        field = self._new_field()
        mol_list = self._mol_list(mol_name)

        descriptor_rows = []

        for xyz_path in mol_list:
            elements, coordinates, charges = (
                self._read_xyz_nbo(xyz_path)
            )

            grid = field.calc_field(
                coordinates,
                elements.tolist(),
                charges
            )

            descriptor_rows.append(
                np.asarray(grid, dtype=float)
            )

        descriptor = np.vstack(descriptor_rows)

        if output_file is not None:
            np.savetxt(
                output_file,
                descriptor,
                fmt=fmt,
                delimiter="\t"
            )
            print(
                f"[DONE] NBO descriptor: {output_file} "
                f"shape={descriptor.shape}"
            )

        return descriptor

    def visualization(
        self,
        coefficient,
        mol_name,
        Directory="visNBO"
    ):
        """
        Generate Mercury overlay XYZ files and detailed TSV files for every
        XYZ listed in mol_name.
        """
        field = self._new_field()
        mol_list = self._mol_list(mol_name)
        intercept, coefficients = (
            self._read_coefficient(coefficient)
        )

        os.makedirs(Directory, exist_ok=True)

        n_voxel = self._n_voxel()

        print(f"[INFO] intercept = {intercept}")
        print(
            f"[INFO] n_voxel = {n_voxel}, "
            f"|coef|>{self.threshold} = "
            f"{np.count_nonzero(np.abs(coefficients) > self.threshold)}"
        )

        results = []

        for xyz_path in mol_list:
            print(f"Processing: {xyz_path}")

            elements, coordinates, charges = (
                self._read_xyz_nbo(xyz_path)
            )

            grid, owner_grid, overlap_count = (
                field.calc_field_full(
                    coordinates,
                    elements.tolist(),
                    charges
                )
            )

            grid = np.asarray(grid, dtype=float)
            owner_grid = np.asarray(owner_grid, dtype=int)
            overlap_count = np.asarray(
                overlap_count,
                dtype=int
            )

            prediction = (
                intercept
                + float(np.dot(grid, coefficients))
            )

            (
                indices,
                categories,
                voxel_xyz,
                selected_coef,
                selected_q,
                selected_owner
            ) = field.extract_voxels(
                grid,
                owner_grid,
                coefficients,
                threshold=self.threshold,
                require_nonzero_q=self.require_nonzero_q,
                eps_q=self.eps_q
            )

            dummy_elements = (
                field.visualization_elements(categories)
            )

            base = os.path.splitext(
                os.path.basename(xyz_path)
            )[0]

            out_xyz = os.path.join(
                Directory,
                f"{base}_overlay.xyz"
            )
            out_tsv = os.path.join(
                Directory,
                f"{base}_overlay_voxels.tsv"
            )

            comment = (
                f"{Directory} | nearest-atom assignment | "
                f"prediction={prediction:.8f} | "
                f"coef!=0 voxels={len(categories)} | "
                f"require_nonzero_q={self.require_nonzero_q}"
            )

            write_overlay_xyz(
                out_xyz,
                elements,
                coordinates,
                dummy_elements,
                voxel_xyz,
                comment=comment
            )

            write_overlay_tsv(
                out_tsv,
                categories,
                voxel_xyz,
                selected_coef,
                selected_q,
                selected_owner,
                elements
            )

            assigned_voxels = int(
                np.count_nonzero(owner_grid >= 0)
            )
            overlap_resolved = int(
                np.count_nonzero(overlap_count >= 2)
            )
            max_overlap = int(overlap_count.max())

            print(
                f"  assigned voxels = "
                f"{assigned_voxels}/{n_voxel}; "
                f"overlap resolved = {overlap_resolved}; "
                f"max overlap = {max_overlap}; "
                f"selected voxels = {len(categories)}; "
                f"prediction = {prediction:.10f}"
            )

            results.append(
                {
                    "input_xyz": str(xyz_path),
                    "overlay_xyz": out_xyz,
                    "overlay_tsv": out_tsv,
                    "prediction": prediction,
                    "selected_voxels": len(categories)
                }
            )

        print(
            f"[DONE] NBO visualization: {Directory}"
        )

        return results
