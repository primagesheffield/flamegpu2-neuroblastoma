#include "header.h"

FLAMEGPU_AGENT_FUNCTION(sc_cell_lifecycle, flamegpu::MessageNone, flamegpu::MessageNone) {
    // @todo 
}
FLAMEGPU_AGENT_FUNCTION(apply_sc_force, flamegpu::MessageNone, flamegpu::MessageNone) {
    // @todo 
}
FLAMEGPU_AGENT_FUNCTION(output_sc_location, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    // @todo 
}
FLAMEGPU_AGENT_FUNCTION(calculate_sc_force, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    // @todo 
}
FLAMEGPU_AGENT_FUNCTION(output_matrix_grid_cell, flamegpu::MessageNone, flamegpu::MessageNone) {
    // @todo 
}

flamegpu::AgentDescription &defineSchwann(flamegpu::ModelDescription& model) {
    auto& sc = model.newAgent("Schwann");
    // Data Layer 0 (integration with imaging biomarkers)
    {
        sc.newVariable<glm::vec3>("xyz");
    }
    // Initial Conditions.
    {
        sc.newVariable<glm::vec3>("Fxyz");
        sc.newVariable<float>("overlap");
        // neighbours is the number of cells within the cell's search distance.
        sc.newVariable<int>("neighbours");
        sc.newVariable<int>("mobile");
        sc.newVariable<int>("ATP");
        sc.newVariable<unsigned int>("cycle");
        sc.newVariable<int>("apop");
        sc.newVariable<int>("apop_signal");
        sc.newVariable<int>("necro");
        sc.newVariable<int>("necro_signal");
        sc.newVariable<int>("necro_critical");
        sc.newVariable<int>("telo_count");
        sc.newVariable<float>("degdiff");
        sc.newVariable<float>("cycdiff");
    }
    // Attribute Layer 1
    {
        sc.newVariable<int>("hypoxia");
        sc.newVariable<int>("nutrient");
        sc.newVariable<int>("DNA_damage");
        sc.newVariable<int>("DNA_unreplicated");
    }
    // Internal
    {
        // This is used to provide the dummy_force reduction.
        sc.newVariable<float>("force_magnitude");
    }
    return sc;
}

void initSchwann(flamegpu::HostAPI &FLAMEGPU) {
    auto SC =  FLAMEGPU.agent("Schwann");
    if (FLAMEGPU.agent("Schwann").count() != 0)
        return;  // SC agents must have been loaded already

    // Env properties required for initialising NB agents
    const float R_tumour = FLAMEGPU.environment.getProperty<float>("R_tumour");
    const float cycle_sc = FLAMEGPU.environment.getProperty<float>("cycle_sc");
    const std::array<unsigned int, 4> cycle_stages = FLAMEGPU.environment.getProperty<unsigned int, 4>("cycle_stages");
    const int apop_sc = FLAMEGPU.environment.getProperty<int>("apop_sc");
    const int apop_signal_sc = FLAMEGPU.environment.getProperty<int>("apop_signal_sc");
    const int necro_sc = FLAMEGPU.environment.getProperty<int>("necro_sc");
    const int necro_signal_sc = FLAMEGPU.environment.getProperty<int>("necro_signal_sc");
    const int telo_count_sc = FLAMEGPU.environment.getProperty<int>("telo_count_sc");

    // Env properties required for calculating agent count
    const float rho_tumour = FLAMEGPU.environment.getProperty<float>("rho_tumour");
    const float V_tumour = FLAMEGPU.environment.getProperty<float>("V_tumour");
    const float cellularity = FLAMEGPU.environment.getProperty<float>("cellularity");
    const float theta_sc = FLAMEGPU.environment.getProperty<float>("theta_sc");

    const unsigned int SC_COUNT = (unsigned int)ceil(rho_tumour * V_tumour * cellularity * theta_sc);

    for (unsigned int i = 0; i < SC_COUNT; ++i) {
        auto agt = SC.newAgent();
        // Data Layer 0 (integration with imaging biomarkers).
        agt.setVariable<glm::vec3>("xyz",
            glm::vec3(-R_tumour + (FLAMEGPU.random.uniform<float>() * 2 * R_tumour),
                -R_tumour + (FLAMEGPU.random.uniform<float>() * 2 * R_tumour),
                -R_tumour + (FLAMEGPU.random.uniform<float>() * 2 * R_tumour)));
        // Initial conditions.
        agt.setVariable<glm::vec3>("Fxyz", glm::vec3(0));
        agt.setVariable<float>("overlap", 0);
        agt.setVariable<int>("neighbours", 0);
        agt.setVariable<int>("mobile", 1);
        agt.setVariable<int>("ATP", 1);
        if (cycle_sc >= 0) {
            agt.setVariable<unsigned int>("cycle", static_cast<unsigned int>(cycle_sc));
        } else {
            // Weird init, because Py model has uniform chance per stage
            // uniform chance within stage
            const int stage = FLAMEGPU.random.uniform<int>(0, 3); // Random int in range [0, 3]
            const unsigned int stage_start = stage == 0 ? 0 : cycle_stages[stage - 1];
            const unsigned int stage_extra = static_cast<unsigned int>(FLAMEGPU.random.uniform<float>() * static_cast<float>(cycle_stages[stage] - stage_start));
            agt.setVariable<unsigned int>("cycle", stage_start + stage_extra);
        }
        agt.setVariable<int>("apop", apop_sc < 0 ? 0 : apop_sc);
        agt.setVariable<int>("apop_signal", apop_signal_sc < 0 ? 0 : apop_signal_sc);
        agt.setVariable<int>("necro", necro_sc < 0 ? 0 : necro_sc);
        agt.setVariable<int>("necro_signal", necro_signal_sc < 0 ? 0 : necro_signal_sc);
        agt.setVariable<int>("telo_count", FLAMEGPU.random.uniform<int>(3, 168)); // Random int in range [3, 168]
        agt.setVariable<int>("telo_count", telo_count_sc < 0 ? FLAMEGPU.random.uniform<int>(25, 35) : telo_count_sc); // Random int in range [25, 35]
        // Attribute Layer 1
        agt.setVariable<int>("hypoxia", 0);
        agt.setVariable<int>("nutrient", 1);
        agt.setVariable<int>("DNA_damage", 0);
        agt.setVariable<int>("DNA_unreplicated", 0);
    }
}