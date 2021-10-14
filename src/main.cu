#define _USE_MATH_DEFINES
#include <cmath>
#include "header.h"
FLAMEGPU_AGENT_FUNCTION(temp_cell_cycle, flamegpu::MessageNone, flamegpu::MessageNone) {
    float age = FLAMEGPU->getVariable<float>("age");
    const float SPEED = FLAMEGPU->environment.getProperty<float>("birth_speed");
    age += SPEED;
    if (age >= 24.0f) {
        // Reset age
        age = FLAMEGPU->random.uniform<float>();
        // birth new agent;
        FLAMEGPU->agent_out.setVariable<float>("age", age);
        FLAMEGPU->agent_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x") + (FLAMEGPU->random.uniform<float>() * 10.0f) - 5.0f);
        FLAMEGPU->agent_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y") + (FLAMEGPU->random.uniform<float>() * 10.0f) - 5.0f);
        FLAMEGPU->agent_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z") + (FLAMEGPU->random.uniform<float>() * 10.0f) - 5.0f);
    }
    FLAMEGPU->setVariable<float>("age", age);
    return flamegpu::ALIVE;
}
FLAMEGPU_CUSTOM_REDUCTION(ReduceMin, a, b) {
    return a < b ? a : b;
}
FLAMEGPU_CUSTOM_REDUCTION(ReduceMax, a, b) {
    return a > b ? a : b;
}
FLAMEGPU_STEP_FUNCTION(logDensity) {
    auto nb = FLAMEGPU->agent("Neuroblastoma");
    auto sc = FLAMEGPU->agent("Schwann");
    // Calculate spherical tumour volume
    const float min_pos_x = std::min<float>(nb.min<float>("x"), sc.min<float>("x"));
    const float min_pos_y = std::min<float>(nb.min<float>("y"), sc.min<float>("y"));
    const float min_pos_z = std::min<float>(nb.min<float>("z"), sc.min<float>("z"));
    const float max_pos_x = std::max<float>(nb.max<float>("x"), sc.max<float>("x"));
    const float max_pos_y = std::max<float>(nb.max<float>("y"), sc.max<float>("y"));
    const float max_pos_z = std::max<float>(nb.max<float>("z"), sc.max<float>("z"));
    const float tumour_volume = 4.0f / 3.0f * M_PI * ((max_pos_x - min_pos_x) / 2.0f) * ((max_pos_y - min_pos_y) / 2.0f) * ((max_pos_z - min_pos_z) / 2.0f);
    // Calculate cells per cubic micron
    const float tumour_density = tumour_volume >  0? (nb.count() + sc.count()) / tumour_volume : 0;
    // 1.91e-3 cells per cubic micron (Louis and Shohet, 2015).
    // In a tumour with limited stroma, and whose cells are small, this number is 1e9 cells/cm3 or 1e-3 cells/cubic micron (Del Monte, 2009).
    // However, because cells have different sizes and shapes, the number can be as low as 3.7e7 cells/cm3 or 3.7e-5 cells/cubic micron in a tumour without extracellular structures (Del Monte, 2009).
    //printf("%u: Volume: %g, Cells: %u, Density %g cells/cubic micron %s\n", FLAMEGPU->getStepCounter(), tumour_volume, nb.count() + sc.count(), tumour_density, tumour_density > 1.91e-3 ? " (Too Dense?)" : "");
    // Markdown table log
    if (FLAMEGPU->getStepCounter() == 0) {
        // Print header
        printf("|Step|Volume|Cells|Density|Note|\n");
        printf("|---|---|---|---|---|\n");
    }
    printf("|%u|%g|%u|%g|%s|\n", FLAMEGPU->getStepCounter(), tumour_volume, nb.count() + sc.count(), tumour_density, tumour_density > 1.91e-3 ? "Too Dense?" : "");
}

FLAMEGPU_INIT_FUNCTION(ModelInit) {
    // With pete's change, these could be split into separate init functions that run in order
    // defineEnviornment procs init env
    initNeuroblastoma(*FLAMEGPU);
    initSchwann(*FLAMEGPU);
    initGrid(*FLAMEGPU);
}


int main(int argc, const char ** argv) {
    const unsigned int CELL_COUNT = 1024;
    flamegpu::ModelDescription model("PRIMAGE: Neuroblastoma");

    // Define environment and agents (THE ORDER HERE IS FIXED, AS THEY ADD INIT FUNCTIONS, ENV MUST COME FIRST)
    defineEnvironment(model, CELL_COUNT);
    flamegpu::AgentDescription& nb = defineNeuroblastoma(model);
    flamegpu::AgentDescription& sc = defineSchwann(model);
    flamegpu::AgentDescription& gc = defineGrid(model);

    auto& nb_cc = nb.newFunction("nb_cell_cycle", temp_cell_cycle);
    nb_cc.setAgentOutput(nb);
    auto& sc_cc = sc.newFunction("sc_cell_cycle", temp_cell_cycle);
    sc_cc.setAgentOutput(sc);

    flamegpu::SubModelDescription& forceResolution = defineForceResolution(model);


    /**
     * Control flow
     */
    {   // Attach init/step/exit functions and exit condition
        model.addInitFunction(ModelInit);
        model.newLayer().addSubModel(forceResolution);
        auto &l2 = model.newLayer();
        l2.addAgentFunction(nb_cc);
        l2.addAgentFunction(sc_cc);
        model.addStepFunction(logDensity);
    }

    /**
     * Create Model Runner
     */
    flamegpu::CUDASimulation cuda_model(model, argc, argv);

    /**
     * Create visualisation
     */
#ifdef VISUALISATION
    flamegpu::visualiser::ModelVis &m_vis = cuda_model.getVisualisation();
    {
        const float INIT_CAM = 1000 * 1.25F;
        // m_vis.setBeginPaused(true);
        m_vis.setInitialCameraLocation(INIT_CAM, INIT_CAM, INIT_CAM);
        m_vis.setCameraSpeed(0.1f);
        auto& nb_agt = m_vis.addAgent("Neuroblastoma");
        auto &sc_agt = m_vis.addAgent("Schwann");
        // Position vars are named x, y, z; so they are used by default
        nb_agt.setModel(flamegpu::visualiser::Stock::Models::ICOSPHERE);
        nb_agt.setModelScale(22); // 2 * env::R_CELL
        sc_agt.setModel(flamegpu::visualiser::Stock::Models::ICOSPHERE);
        sc_agt.setModelScale(22); // 2 * env::R_CELL
        // Render the messaging bounding box, -1000 - 1000 each dimension
        {
            auto pen = m_vis.newLineSketch(1, 1, 1, 0.2f);  // white
            // X lines
            pen.addVertex(-1000,  1000,  1000);
            pen.addVertex( 1000,  1000,  1000);
            pen.addVertex(-1000,  1000, -1000);
            pen.addVertex( 1000,  1000, -1000);
            pen.addVertex(-1000, -1000,  1000);
            pen.addVertex( 1000, -1000,  1000);
            pen.addVertex(-1000, -1000, -1000);
            pen.addVertex( 1000, -1000, -1000);
            // Y lines
            pen.addVertex( 1000, -1000,  1000);
            pen.addVertex( 1000,  1000,  1000);
            pen.addVertex( 1000, -1000, -1000);
            pen.addVertex( 1000,  1000, -1000);
            pen.addVertex(-1000, -1000,  1000);
            pen.addVertex(-1000,  1000,  1000);
            pen.addVertex(-1000, -1000, -1000);
            pen.addVertex(-1000,  1000, -1000);
            // Z lines
            pen.addVertex( 1000,  1000, -1000);
            pen.addVertex( 1000,  1000,  1000);
            pen.addVertex( 1000, -1000, -1000);
            pen.addVertex( 1000, -1000,  1000);
            pen.addVertex(-1000,  1000, -1000);
            pen.addVertex(-1000,  1000,  1000);
            pen.addVertex(-1000, -1000, -1000);
            pen.addVertex(-1000, -1000,  1000);
        }
    }
    m_vis.activate();
#endif
    /**
     * Initialisation
     */
    if (cuda_model.getSimulationConfig().input_file.empty()) {
        // Not sure if a default init is necessary yet
    }

    /**
     * Execution
     */
    cuda_model.simulate();

#ifdef VISUALISATION
    m_vis.join();
#endif
    return 0;
}
