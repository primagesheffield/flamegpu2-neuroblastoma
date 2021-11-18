#include "header.h"

FLAMEGPU_INIT_FUNCTION(ModelInit) {
    // With pete's change, these could be split into separate init functions that run in order
    // defineEnvironment procs init env
    initNeuroblastoma(*FLAMEGPU);
    initSchwann(*FLAMEGPU);
    initGrid(*FLAMEGPU);
}
FLAMEGPU_STEP_FUNCTION(validation) {
    auto d = FLAMEGPU->environment.getMacroProperty<unsigned int, 16>("sc_cycle");
    std::array<unsigned int, 16> rtn;
    for (int i = 0; i < 16; ++i)
        rtn[i] = d[i];
    FLAMEGPU->environment.setProperty("sc_cycle_stage", rtn);
    d.zero();
}

void defineModel(flamegpu::ModelDescription& model) {
    // Define environment and agents (THE ORDER HERE IS FIXED, AS THEY ADD INIT FUNCTIONS, ENV MUST COME FIRST)
    defineEnvironment(model);
    const flamegpu::AgentDescription& nb = defineNeuroblastoma(model);
    const flamegpu::AgentDescription& sc = defineSchwann(model);
    const flamegpu::AgentDescription& gc = defineGrid(model);

    const flamegpu::SubModelDescription& forceResolution = defineForceResolution(model);

    /**
     * Control flow
     */
    {   // Attach init/step/exit functions and exit condition
        model.addInitFunction(ModelInit);
        // Force resolution
        model.newLayer().addSubModel(forceResolution);
        // Expand grid
        model.newLayer().addHostFunction(CAexpand);
        // Reset grid
        model.newLayer().addHostFunction(reset_grids);
        // Vasculature
        model.newLayer().addHostFunction(vasculature);
        // Output oxygen/matrix grid
        auto& l_output_grid = model.newLayer();
        l_output_grid.addAgentFunction(nb.getFunction("output_oxygen_cell"));
        l_output_grid.addAgentFunction(sc.getFunction("output_matrix_grid_cell"));
        // Alter
        model.newLayer().addAgentFunction(gc.getFunction("alter"));
        model.newLayer().addHostFunction(alter2);
        // Cell cycle
        auto& l_cycle = model.newLayer();
        l_cycle.addAgentFunction(nb.getFunction("nb_cell_lifecycle"));
        l_cycle.addAgentFunction(sc.getFunction("sc_cell_lifecycle"));
        model.newLayer().addHostFunction(validation);
    }
}
#ifdef VISUALISATION
flamegpu::visualiser::ModelVis& defineVisualisation(flamegpu::ModelDescription& model, flamegpu::CUDASimulation& sim) {
    flamegpu::visualiser::ModelVis& m_vis = sim.getVisualisation();
    {
        const float INIT_CAM = 1000 * 1.25F;
        // m_vis.setBeginPaused(true);
        m_vis.setInitialCameraLocation(INIT_CAM, INIT_CAM, INIT_CAM);
        m_vis.setCameraSpeed(0.1f);
        m_vis.setViewClips(10.0f, 6000.0f);
        auto& nb_agt = m_vis.addAgent("Neuroblastoma");
        auto& sc_agt = m_vis.addAgent("Schwann");
        nb_agt.setXYZVariable("xyz");
        nb_agt.setModel(flamegpu::visualiser::Stock::Models::ICOSPHERE);
        nb_agt.setModelScale(model.getEnvironment().getProperty<float>("R_cell") * 2.0f); // Could improve this in future to use the dynamic rad
        sc_agt.setXYZVariable("xyz");
        sc_agt.setModel(flamegpu::visualiser::Stock::Models::ICOSPHERE);
        sc_agt.setModelScale(model.getEnvironment().getProperty<float>("R_cell") * 2.0f); // Could improve this in future to use the dynamic rad
        // Render the messaging bounding box, -1000 - 1000 each dimension
        {
            auto pen = m_vis.newLineSketch(1, 1, 1, 0.2f);  // white
            // X lines
            pen.addVertex(-2000, 2000, 2000);
            pen.addVertex(2000, 2000, 2000);
            pen.addVertex(-2000, 2000, -2000);
            pen.addVertex(2000, 2000, -2000);
            pen.addVertex(-2000, -2000, 2000);
            pen.addVertex(2000, -2000, 2000);
            pen.addVertex(-2000, -2000, -2000);
            pen.addVertex(2000, -2000, -2000);
            // Y lines
            pen.addVertex(2000, -2000, 2000);
            pen.addVertex(2000, 2000, 2000);
            pen.addVertex(2000, -2000, -2000);
            pen.addVertex(2000, 2000, -2000);
            pen.addVertex(-2000, -2000, 2000);
            pen.addVertex(-2000, 2000, 2000);
            pen.addVertex(-2000, -2000, -2000);
            pen.addVertex(-2000, 2000, -2000);
            // Z lines
            pen.addVertex(2000, 2000, -2000);
            pen.addVertex(2000, 2000, 2000);
            pen.addVertex(2000, -2000, -2000);
            pen.addVertex(2000, -2000, 2000);
            pen.addVertex(-2000, 2000, -2000);
            pen.addVertex(-2000, 2000, 2000);
            pen.addVertex(-2000, -2000, -2000);
            pen.addVertex(-2000, -2000, 2000);
        }
    }
    m_vis.activate();
    return m_vis;
}
#endif
