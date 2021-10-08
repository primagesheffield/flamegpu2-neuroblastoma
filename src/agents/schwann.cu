#include "header.h"

flamegpu::AgentDescription &defineSchwann(flamegpu::ModelDescription& model) {
    auto& sc = model.newAgent("Schwann");
    // Data Layer 0 (integration with imaging biomarkers)
    {
        sc.newVariable<float>("x");
        sc.newVariable<float>("y");
        sc.newVariable<float>("z");
    }
    // Initial Conditions.
    {
        sc.newVariable<float>("Fx");
        sc.newVariable<float>("Fy");
        sc.newVariable<float>("Fz");
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
