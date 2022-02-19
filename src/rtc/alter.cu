#define GRID_MAX_DIMENSIONS 151
#define GMD GRID_MAX_DIMENSIONS
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
    float matrix_value = matrix_grid[location.x][location.y][location.z].exchange(0);  // This is required to read+write in same fn
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

    return flamegpu::ALIVE;
}