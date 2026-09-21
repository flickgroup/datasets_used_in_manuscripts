/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(cboamd,FixCBOAMD);
// clang-format on
#else

#ifndef LMP_FIX_CBOAMD_H
#define LMP_FIX_CBOAMD_H

#include "fix.h"

// DeePMD C++ interface includes
#include "deepmd/DeepPot.h"
#include "deepmd/DeepDipole.h"
#include "deepmd/DeepPolar.h"

namespace LAMMPS_NS {

class FixCBOAMD : public Fix {
 public:
  FixCBOAMD(class LAMMPS *, int, char **);
  ~FixCBOAMD() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void post_force(int) override;
  void post_force_respa(int, int, int) override;
  double compute_scalar() override;
  double compute_vector(int) override;
  double compute_array(int, int) override;
  void write_restart(FILE *) override;
  void restart(char *) override;
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  void set_arrays(int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  int pack_restart(int, double *) override;
  int unpack_restart(double *) override;
  int size_restart(int) override;
  int maxsize_restart() override;

 private:
  // DeepMD model paths
  char *model_potential;
  char *model_dipole;
  char *model_polar;
  
  // DeepMD model objects using C++ interface
  deepmd::DeepPot *deepmd_potential;
  deepmd::DeepDipole *deepmd_dipole;
  deepmd::DeepPolar *deepmd_polar;
  
  // Type mapping for atoms
  int *type_map;
  int ntypes;
  
  // Output arrays
  double *dipole;
  double *polarizability;
  double *forces_deepmd;
  
  // DeepMD computation arrays
  std::vector<double> coords_deepmd;
  std::vector<int> atom_types_deepmd;
  std::vector<double> cell_deepmd;
  std::vector<double> energy_deepmd;
  std::vector<double> forces_deepmd_vec;
  std::vector<double> dipole_deepmd;
  std::vector<double> dipole_grad_deepmd;
  std::vector<double> polar_deepmd;
  std::vector<double> polar_grad_deepmd;
  
  // Cavity Born-Oppenheimer parameters
  bool photons_enabled;
  int nphoton;
  double *omega_photon;
  double *lambda_photon;
  double **lambda_vector;
  
  // Photon coordinates and momenta
  double *qa, *pa, *fa, *ea;
  
  // Time step
  double dt;
  
  // Current step
  int current_step;
  
  // Output file
  FILE *output_file;
  
  // Helper functions
  void init_deepmd_models();
  void cleanup_deepmd_models();
  void compute_deepmd_energy_forces();
  void compute_deepmd_dipole();
  void compute_deepmd_polarizability();
  void update_photon_coordinates();
  void compute_cboa_forces();
  void write_output();
  
  // DeepMD computation helpers
  void convert_coordinates_to_deepmd_format();
  void convert_forces_from_deepmd_format();
  
  // Unit conversion constants
  static constexpr double EV_TO_HARTREE = 0.0367493;
  static constexpr double ANGSTROM_TO_BOHR = 1.88973;
  static constexpr double EV_PER_ANGSTROM_TO_HARTREE_PER_BOHR = 0.0194467;
};

}

#endif
#endif
