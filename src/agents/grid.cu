#include "header.h"

// @todo init_grid?
FLAMEGPU_AGENT_FUNCTION(alter, flamegpu::MessageNone, flamegpu::MessageNone) {
    // @todo 
}
FLAMEGPU_AGENT_FUNCTION(fresolve_CAexpand_device, flamegpu::MessageNone, flamegpu::MessageNone) {
    // @todo 
}

flamegpu::AgentDescription &defineGrid(flamegpu::ModelDescription& model) {
    auto& gc = model.newAgent("GridCell");
    // Cell coordinate
    {
        gc.newVariable<float>("x");
        gc.newVariable<float>("y");
        gc.newVariable<float>("z");
    }
    // Properties
    {
        gc.newVariable<unsigned int>("Nnbl_grid");
        gc.newVariable<unsigned int>("Nscl_grid");
        gc.newVariable<unsigned int>("N_l_grid");
        gc.newVariable<unsigned int>("matrix_value");
        gc.newVariable<unsigned int>("N_grid");
    }
    // Nnb and Nsc are used by CAexpand and to calculate d_N_grid_VolEx, this includes migration_CAexpand_device() and alter() after force-resolution submodel
    // They are incremented by migrate_nb(), migrate_sc(), output_oxygen_cell()[and init version], output_matrix_grid_cell()[and init version] respectively
    // They are consumed by migration_CAexpand_device() and alter() to calculate d_N_grid_VolEx
    // They are only reset by migration_CAexpand_device()
    // alter() does not reset these values, as cell_cycle() changes them to reflect births/deaths, ready for migration submodel
    model.Environment().newMacroProperty<unsigned int, GMD, GMD, GMD>("Nnb_grid");
    model.Environment().newMacroProperty<unsigned int, GMD, GMD, GMD>("Nsc_grid");
    // Nsca, Nnba are not yet used, they will however be required outside of the migration submodel
    model.Environment().newMacroProperty<unsigned int, GMD, GMD, GMD>("Nsca_grid");  // Location of all SC cells with apop == 1
    model.Environment().newMacroProperty<unsigned int, GMD, GMD, GMD>("Nnba_grid");  // Location of all NB cells with apop == 1
    // Nnbn and Nscn, are consumed by Neuroblastoma_sense() and Schwann_CellLifecycle()
    // They are incremented by output_oxygen_cell and output_matrix_grid_cell
    // They are reset by CAexpand and fresolve_CAexpand_device
    // There is a slight issue where they hold the wrong values through the first step of force resolution
    // However they are not currently read there, so this should not be a problem
    model.Environment().newMacroProperty<unsigned int, GMD, GMD, GMD>("Nnbn_grid");
    model.Environment().newMacroProperty<unsigned int, GMD, GMD, GMD>("Nscn_grid");
    // Nnbl and Nscl are consumed by alter() for calculating oxygen consumption
    // They are also consumed by Neuroblastoma_sense() and vasculature()
    // They are incremented by output_oxygen_cell and output_matrix_grid_cell
    // They are reset by CAexpand and fresolve_CAexpand_device
    // There is a slight issue where they hold the wrong values through the first step of force resolution
    // However they are not currently read there, so this should not be a problem
    model.Environment().newMacroProperty<unsigned int, GMD, GMD, GMD>("Nnbl_grid");
    model.Environment().newMacroProperty<unsigned int, GMD, GMD, GMD>("Nscl_grid");
    // Similar to above, but consumed by alter()
    model.Environment().newMacroProperty<unsigned int, GMD, GMD, GMD>("Nscl_col_grid");
    // N_grid is used for calculating N_grid_VolEx for CellCycle and Migration, it is the sum of d_Nnb_grid and d_Nsc_grid
    // This isn't really required, it's mostly a persistent track of the number of cells per grid, used for density logging
    // It isn't currently updated by cell division/death during cell_cycle
    model.Environment().newMacroProperty<unsigned int, GMD, GMD, GMD>("N_grid");
    // Calculated by alter() with O2_grid, although we don't represent O2 as a grid
    model.Environment().newMacroProperty<float, GMD, GMD, GMD>("matrix_grid");
    return gc;
}

void initGrid(flamegpu::HostAPI &FLAMEGPU) {
    auto gc =  FLAMEGPU.agent("GridCell");
    if (FLAMEGPU.agent("GridCell").count() != 0)
        return;  // gc agents must have been loaded already

    const float cellularity = FLAMEGPU.environment.getProperty<float>("cellularity");
    const unsigned int GC_COUNT = (unsigned int)pow(GRID_MAX_DIMENSIONS, 3);

    for (unsigned int i = 0; i < GC_COUNT; ++i) {
        auto agt = gc.newAgent();
        agt.setVariable<unsigned int>("x", i % GRID_MAX_DIMENSIONS);
        agt.setVariable<unsigned int>("y", (i / GRID_MAX_DIMENSIONS) % GRID_MAX_DIMENSIONS);
        agt.setVariable<unsigned int>("z", i / (GRID_MAX_DIMENSIONS * GRID_MAX_DIMENSIONS));
        agt.setVariable<unsigned int>("Nnbl_grid", 0);
        agt.setVariable<unsigned int>("Nscl_grid", 0);
        agt.setVariable<unsigned int>("N_l_grid", 0);
        agt.setVariable<float>("matrix_value", 1.0f - cellularity);
        agt.setVariable<float>("N_grid", cellularity);
    }
    for (unsigned int x = 0; x < GRID_MAX_DIMENSIONS; ++x) {
        for (unsigned int y = 0; y < GRID_MAX_DIMENSIONS; ++y) {
            for (unsigned int z = 0; z < GRID_MAX_DIMENSIONS; ++z) {
                FLAMEGPU.environment.getMacroProperty<float, GMD, GMD, GMD>("matrix_grid")[x][y][z] = 1.0f - cellularity;
                // Other values all default init to 0
            }
        }
    }
}
