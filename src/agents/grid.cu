#include "header.h"

FLAMEGPU_AGENT_FUNCTION(alter, flamegpu::MessageNone, flamegpu::MessageNone) {
    const glm::uvec3 location = FLAMEGPU->getVariable<glm::uvec3>("xyz");
    const glm::uvec3 grid_origin = FLAMEGPU->environment.getProperty<glm::uvec3>("grid_origin");
    // Skip inactive agents
    if (location.x < grid_origin.x || location.x >= GMD - grid_origin.x ||
        location.y < grid_origin.y || location.y >= GMD - grid_origin.y ||
        location.z < grid_origin.z || location.z >= GMD - grid_origin.z) {
        FLAMEGPU->setVariable<unsigned int>("Nnbl_grid", 0);
        FLAMEGPU->setVariable<unsigned int>("Nscl_grid", 0);
        FLAMEGPU->setVariable<unsigned int>("N_l_grid", 0);
        FLAMEGPU->setVariable<float>("matrix_value", 0);
        FLAMEGPU->setVariable<unsigned int>("N_grid", 0);
        return flamegpu::ALIVE;
    }

    const auto matrix_grid = FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("matrix_grid");
    float matrix_value = matrix_grid[location.x][location.y][location.z].exchange(0); // This is required to read+write in same fn
    const glm::uvec3 grid_span = FLAMEGPU->environment.getProperty<glm::uvec3>("grid_span");
    const glm::uvec3 grid_span_old = FLAMEGPU->environment.getProperty<glm::uvec3>("grid_span_old");
    // Apply CAexpand, if required, to grid members
    if (grid_span != grid_span_old) {
        // matrix grid size has changed
        // If we are in the new area
        if ((location.z < grid_origin.z + grid_span.z - grid_span_old.z) ||
            (location.z > grid_origin.z + grid_span.z - 1 + grid_span_old.z) ||
            (location.y > grid_origin.y + grid_span.y - grid_span_old.y) ||
            (location.y < grid_origin.y + grid_span.y - 1 + grid_span_old.y) ||
            (location.x > grid_origin.x + grid_span.x - grid_span_old.x) ||
            (location.x < grid_origin.x + grid_span.x - 1 + grid_span_old.x)) {
            matrix_value = FLAMEGPU->environment.getProperty<float>("matrix_dummy");
        }
    }

    // Update the extracellular environment.
    // 1. Oxygen level(if it is not static) :
    // Calculate the amount consumed by living neuroblasts and Schwann cells in a time step.
    // Add the amount supplied by the vasculature.
    // Assuming a diffusivity of 1.75e-5 cm2 s - 1 (Grote et al., 1977), the diffusion length in an hour is 0.5 cm or 5 mm.Diffusion is not limiting.
    // 2. Matrix volume fraction :
    // Calculate the volume produced by living Schwann cells in a time step.
    //
    // Sum these grid values, so that alter2() can perform a reduction for oxygen calc
    const auto Nnbl_grid = FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nnbl_grid");
    const auto Nscl_grid = FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nscl_grid");
    unsigned int s_Nnbl_grid = Nnbl_grid[location.x][location.y][location.z];
    unsigned int s_Nscl_grid = Nscl_grid[location.x][location.y][location.z];
    FLAMEGPU->setVariable<unsigned int>("Nnbl_grid", s_Nnbl_grid);
    FLAMEGPU->setVariable<unsigned int>("Nscl_grid", s_Nscl_grid);
    FLAMEGPU->setVariable<unsigned int>("N_l_grid", s_Nnbl_grid + s_Nscl_grid);  // This is kind of redundant, could reduce and sum both vals
    const int SKIP_ALTER = FLAMEGPU->environment.getProperty<int>("SKIP_ALTER");
    // Don't update O2/matrix if it's INIT pass
    if (!SKIP_ALTER) {
        const float P_matrix = FLAMEGPU->environment.getProperty<float>("P_matrix");
        const unsigned int step_size = FLAMEGPU->environment.getProperty<unsigned int>("step_size");
        const float V_grid = FLAMEGPU->environment.getProperty<float>("V_grid");
        const auto Nscl_col_grid = FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nscl_col_grid");
        // Update our matrix value
        matrix_value += Nscl_col_grid[location.x][location.y][location.z] * P_matrix * step_size / V_grid;
        FLAMEGPU->setVariable<float>("matrix_value", matrix_value);
    }
    matrix_grid[location.x][location.y][location.z].exchange(matrix_value);

    const auto Nnb_grid = FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nnb_grid");
    const auto Nsc_grid = FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nsc_grid");
    const unsigned int s_N_grid = Nnb_grid[location.x][location.y][location.z] + Nsc_grid[location.x][location.y][location.z];
    FLAMEGPU->setVariable<unsigned int>("N_grid", s_N_grid);
    auto N_grid = FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("N_grid");
    N_grid[location.x][location.y][location.z].exchange(s_N_grid);
    // Nsca_grid[i->x][i->y][i->z] = 0;
    // Nnba_grid[i->x][i->y][i->z] = 0;
    // Nnbl_grid[i->x][i->y][i->z] = 0;
    // Nscl_grid[i->x][i->y][i->z] = 0;
    // Nscl_col_grid[i->x][i->y][i->z] = 0;
    // Nnb_grid, d_Nsc_grid, Counts are not reset here, as they are updated by cell cycle
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(fresolve_CAexpand_device, flamegpu::MessageNone, flamegpu::MessageNone) {
    const glm::uvec3 location = FLAMEGPU->getVariable<glm::uvec3>("xyz");
    const glm::uvec3 grid_origin = FLAMEGPU->environment.getProperty<glm::uvec3>("grid_origin");
    const glm::uvec3 grid_dims = FLAMEGPU->environment.getProperty<glm::uvec3>("grid_dims");  // GRID_MAX_DIMENSIONS
    // Skip inactive agents
    if (location.x < grid_origin.x || location.x >= grid_dims.x - grid_origin.x ||
        location.y < grid_origin.y || location.y >= grid_dims.y - grid_origin.y ||
        location.z < grid_origin.z || location.z >= grid_dims.z - grid_origin.z) {
        FLAMEGPU->setVariable<unsigned int>("Nnbl_grid", 0);
        FLAMEGPU->setVariable<unsigned int>("Nscl_grid", 0);
        FLAMEGPU->setVariable<unsigned int>("N_l_grid", 0);
        FLAMEGPU->setVariable<float>("matrix_value", 0);
        FLAMEGPU->setVariable<unsigned int>("N_grid", 0);
        return flamegpu::ALIVE;
    }
    const auto matrix_grid = FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("matrix_grid");
    float matrix_value = matrix_grid[location.x][location.y][location.z].exchange(0); // This is required to read+write in same fn
    const glm::uvec3 grid_span = FLAMEGPU->environment.getProperty<glm::uvec3>("grid_span");
    const glm::uvec3 grid_span_old = FLAMEGPU->environment.getProperty<glm::uvec3>("grid_span_old");
    // Apply CAexpand, if required, to grid members
    if (grid_span != grid_span_old) {
        // matrix grid size has changed
        // If we are in the new area
        if ((location.z < grid_origin.z + grid_span.z - grid_span_old.z) ||
            (location.z > grid_origin.z + grid_span.z - 1 + grid_span_old.z) ||
            (location.y > grid_origin.y + grid_span.y - grid_span_old.y) ||
            (location.y < grid_origin.y + grid_span.y - 1 + grid_span_old.y) ||
            (location.x > grid_origin.x + grid_span.x - grid_span_old.x) ||
            (location.x < grid_origin.x + grid_span.x - 1 + grid_span_old.x)) {
            matrix_value = FLAMEGPU->environment.getProperty<float>("matrix_dummy");
        }
    }
    auto Nnbl_grid = FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nnbl_grid");
    auto Nscl_grid = FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nscl_grid");
    unsigned int s_Nnbl_grid = Nnbl_grid[location.x][location.y][location.z].exchange(0);  // These reset here during force resolution;
    unsigned int s_Nscl_grid = Nscl_grid[location.x][location.y][location.z].exchange(0);  // These reset here during force resolution;
    FLAMEGPU->setVariable<unsigned int>("Nnbl_grid", s_Nnbl_grid);
    FLAMEGPU->setVariable<unsigned int>("Nscl_grid", s_Nscl_grid);
    FLAMEGPU->setVariable<unsigned int>("N_l_grid", s_Nnbl_grid + s_Nscl_grid);  // This is kind of redundant, could reduce and sum both vals
    const auto Nnb_grid = FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nnb_grid");
    const auto Nsc_grid = FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nsc_grid");
    // Count has been consumed, reset to 0
    const unsigned int s_N_grid = Nnb_grid[location.x][location.y][location.z].exchange(0) + Nsc_grid[location.x][location.y][location.z].exchange(0);
    FLAMEGPU->setVariable<unsigned int>("N_grid", s_N_grid);
    auto N_grid = FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("N_grid");
    N_grid[location.x][location.y][location.z].exchange(s_N_grid);

    // These reset here during force resolution
    FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nscl_col_grid").exchange(0);
    FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nnbn_grid").exchange(0);
    FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nscn_grid").exchange(0);
    return flamegpu::ALIVE;
}

flamegpu::AgentDescription &defineGrid(flamegpu::ModelDescription& model) {
    auto& gc = model.newAgent("GridCell");
    // Cell coordinate
    {
        gc.newVariable<unsigned int, 3>("xyz");
    }
    // Properties
    {
        gc.newVariable<unsigned int>("Nnbl_grid");
        gc.newVariable<unsigned int>("Nscl_grid");
        gc.newVariable<unsigned int>("N_l_grid");
        gc.newVariable<float>("matrix_value");
        gc.newVariable<unsigned int>("N_grid");
    }
    // Agent Functions
    {
        gc.newFunction("alter", alter);
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
        agt.setVariable<unsigned int, 3>("xyz", {
            i % GRID_MAX_DIMENSIONS, 
            (i / GRID_MAX_DIMENSIONS) % GRID_MAX_DIMENSIONS,
            i / (GRID_MAX_DIMENSIONS * GRID_MAX_DIMENSIONS)
        });
        agt.setVariable<unsigned int>("Nnbl_grid", 0);
        agt.setVariable<unsigned int>("Nscl_grid", 0);
        agt.setVariable<unsigned int>("N_l_grid", 0);
        agt.setVariable<float>("matrix_value", 1.0f - cellularity);
        agt.setVariable<unsigned int>("N_grid", 0);
    }
    auto matrix_grid = FLAMEGPU.environment.getMacroProperty<float, GMD, GMD, GMD>("matrix_grid");
    for (unsigned int x = 0; x < GRID_MAX_DIMENSIONS; ++x) {
        for (unsigned int y = 0; y < GRID_MAX_DIMENSIONS; ++y) {
            for (unsigned int z = 0; z < GRID_MAX_DIMENSIONS; ++z) {
                matrix_grid[x][y][z] = 1.0f - cellularity;
                // Other values all default init to 0
            }
        }
    }
    // Perform CAExpand to init grid environment properties
    // Except we can't run that now, as agent's don't yet exist on device.
    // CAexpand(&FLAMEGPU);
    const glm::uvec3 oldspan = FLAMEGPU.environment.getProperty<glm::uvec3>("grid_span");
    FLAMEGPU.environment.setProperty<glm::uvec3>("grid_span_old", oldspan);
    const float R_voxel = FLAMEGPU.environment.getProperty<float>("R_voxel");
    const float R_tumour = FLAMEGPU.environment.getProperty<float>("R_tumour");
    glm::uvec3 newspan = glm::uvec3(static_cast<unsigned int>(glm::ceil((R_tumour / R_voxel / 2.0f) + 0.5f)));
    // clamp span (i don't think the python algorithm lets o2 grid shrink)
    const glm::uvec3 newspan_u = max(oldspan, newspan);
    FLAMEGPU.environment.setProperty<glm::uvec3>("grid_span", newspan_u);
    const glm::uvec3 new_grid_dims = (newspan_u * 2u) - glm::uvec3(1);
    FLAMEGPU.environment.setProperty<glm::uvec3>("grid_dims", new_grid_dims);
    if (new_grid_dims.x > GRID_MAX_DIMENSIONS || new_grid_dims.y > GRID_MAX_DIMENSIONS || new_grid_dims.z > GRID_MAX_DIMENSIONS) {
        fprintf(stderr, "grid has grown too large (%u, %u, %u), recompile with bigger!\n", new_grid_dims.x, new_grid_dims.y, new_grid_dims.z);
        throw std::runtime_error("grid has grown too large, recompile with bigger\n");
    }
    glm::uvec3 grid_origin = (glm::uvec3(GRID_MAX_DIMENSIONS) - new_grid_dims) / 2u;
    FLAMEGPU.environment.setProperty<glm::uvec3>("grid_origin", grid_origin);
}
