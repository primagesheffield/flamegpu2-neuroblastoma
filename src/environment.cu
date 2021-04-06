#include "header.h"
/**
 * integration with imaging biomarkers
 * @todo This is unfinished, need to consider how RNG components will be seeded
 */
void data_layer_0(ModelDescription& model) {
    auto& env = model.Environment();
    // Initial volume of tumour (cubic micron).
    env.newProperty<float>("V_tumour", 2e6);
    // Factor by which the tumour can expand in all three directions.
    // According to Aherne and Buck (1971), the volume doubling time is around 60 days.
    // Note that this is illustrative and unrealistic.
    // The orchestrator must provide more details about the boundary geometry, potentially different parameters for different directions.
    env.newProperty<float>("boundary_max", 1.587);
    // Sensitivity of the boundary displacement field to invasion into the boundary
    // Note that this is illustrative and unrealistic.
    // The orchestrator must provide the parameters of a displacement function.
    env.newProperty<float>("x_displace", 2);
    env.newProperty<float>("y_displace", 2);
    env.newProperty<float>("z_displace", 2);
    // Half of a voxel's side length in microns.
    // This must be at least as big as R_cell.
    env.newProperty<float>("R_voxel", 15);
    // Histology type (0 is neuroblastoma, 1 is ganglioneuroblastoma, 2 is nodular ganglioneuroblastoma, 3 is intermixed ganglioneuroblastoma, 4 is ganglioneuroma, 5 is maturing ganglioneuroma, and 6 is mature ganglioneuroma).
    // If it is a ganglioneuroblastoma or a ganglioneuroma, assign the subtype stochastically.
    env.newProperty<int>("histology_init", 0);
    // @todo histology: Derived values should be handled later, how to rng
    // Grade of differentiation for neuroblastoma (0 is undifferentiated, 1 is pooly differentiated, and 2 is differentiating).
    // @todo gradiff: how to rng init env values?
    // Initial cell density of tumour (cells per cubic micron).
    // 1.91e-3 cells per cubic micron (Louis and Shohet, 2015).
    // In a tumour with limited stroma, and whose cells are small, this number is 1e9 cells/cm3 or 1e-3 cells/cubic micron (Del Monte, 2009).
    // However, because cells have different sizes and shapes, the number can be as low as 3.7e7 cells/cm3 in a tumour without extracellular structures (Del Monte, 2009).
    env.newProperty<float>("rho_tumour", 9.39e-05);
    // Cell radius.
    // The radii of most animal cells range from 5 to 15 microns (Del Monte, 2009).
    env.newProperty<float>("R_cell", 11);
    // Initial oxygen level (continuous, 0 to 1).
    // Oxygen concentration in the kidney = 72 mmHg (Carreau et al., 2011), chosen as concentration scale.
    // Oxygen concentration in hypoxic tumours = 2 to 32 mmHg (McKeown, 2014).
    // @todo O2: how to rng init env values?
    // Initial cellularity in the tumour (continuous, 0 to 1).
    // @todo cellularity: Derived values should be handled later, dependent on histology/rng
    // Fraction of Schwann cells in the cell population (continuous, 0 to 1).
    // @todo theta_sc: Derived values should be handled later, dependent on histology/rng

    // Derived values    
}
void mechanical_model_parameters(ModelDescription& model) {
    auto& env = model.Environment();
    // Minimum overlap below which two cells cannot interact, given in m (Pathmanathan et al., 2009), converted to microns.
    // However, it was decided that it should be reset to zero so that when two cells are just touching, they stop interacting immediately, i.e. no bouncing off.
    env.newProperty<float>("min_overlap", -4e-6 * 1e6 * 0);
    // Linear force law parameter in N m-1 (Pathmanathan et al., 2009).
    env.newProperty<float>("k1", 2.2e-3);
    // Critical overlap above which the exponential force law overrides the linear one, given in m (Pathmanathan et al., 2009), converted to microns.
    env.newProperty<float>("crit_overlap", 2e-6 * 1e6);
    // Dimensionless exponential force law parameter (Pathmanathan et al., 2009).
    env.newProperty<float>("alpha", 1);
    // Viscosity given in N s m-1 (Pathmanathan et al., 2009).
    env.newProperty<float>("mu", 0.4);
    // Maximum time step for quasi-steady evolution of tissue mechanics, given in hours (Pathmanathan et al., 2009), converted to seconds.
    env.newProperty<float>("dt_max", 36);
    // Convergence threshold for equations of motion in percent.
    env.newProperty<float>("thres_converge", 10);
}
#define _USE_MATH_DEFINES
#include <math.h>
void defineEnvironment(ModelDescription& model, unsigned int CELL_COUNT) {
    // data_layer_0(model);
    mechanical_model_parameters(model);
    // Temporary additional environment components required for force resolution submodel
    auto& env = model.Environment();
    env.newProperty<float>("rho_tumour", 9.39e-05);
    env.newProperty<float>("cellularity", 0.5f);  // anywhere in range 0.5-1 depending on histology/gradiff
    env.newProperty<float>("theta_sc", 0.5f);  // anywhere in range 0-1 depending on histology/gradiff
    env.newProperty<float>("R_cell", 11);
    env.newProperty<float>("V_tumour", ((float)CELL_COUNT)/ (env.getProperty<float>("rho_tumour") * env.getProperty<float>("cellularity")));
    env.newProperty<float>("R_tumour", (float)cbrt(env.getProperty<float>("V_tumour") * 0.75 / M_PI));
}
