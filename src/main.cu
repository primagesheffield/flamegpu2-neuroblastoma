#include <glm/ext/scalar_constants.hpp>

#include "header.h"

FLAMEGPU_CUSTOM_REDUCTION(glm_min2, a, b) {
    return glm::min(a, b);
}
FLAMEGPU_CUSTOM_REDUCTION(glm_max2, a, b) {
    return glm::max(a, b);
}
FLAMEGPU_STEP_FUNCTION(logDensity) {
    auto nb = FLAMEGPU->agent("Neuroblastoma");
    auto sc = FLAMEGPU->agent("Schwann");
    // Calculate spherical tumour volume
    const glm::vec3 min_pos = min(
        nb.reduce<glm::vec3>("xyz", glm_min2, glm::vec3(std::numeric_limits<float>().max())),
        sc.reduce<glm::vec3>("xyz", glm_min2, glm::vec3(std::numeric_limits<float>().max())));
    const glm::vec3 max_pos = max(
        nb.reduce<glm::vec3>("xyz", glm_max2, glm::vec3(-std::numeric_limits<float>().max())),
        sc.reduce<glm::vec3>("xyz", glm_max2, glm::vec3(-std::numeric_limits<float>().max())));
    const float tumour_volume = 4.0f / 3.0f * glm::pi<float>() * ((max_pos.x - min_pos.x) / 2.0f) * ((max_pos.y - min_pos.y) / 2.0f) * ((max_pos.z - min_pos.z) / 2.0f);
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
    // defineEnvironment procs init env
    initNeuroblastoma(*FLAMEGPU);
    initSchwann(*FLAMEGPU);
    initGrid(*FLAMEGPU);
}


int main(int argc, const char ** argv) {
    const unsigned int CELL_COUNT = 1024;
    flamegpu::ModelDescription model("PRIMAGE: Neuroblastoma");

    // Define environment and agents (THE ORDER HERE IS FIXED, AS THEY ADD INIT FUNCTIONS, ENV MUST COME FIRST)
    defineEnvironment(model, CELL_COUNT);
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
        // Reset grid
        model.newLayer().addHostFunction(reset_grids);
        // Output oxygen/matrix grid
        auto& l_output_grid = model.newLayer();
        l_output_grid.addAgentFunction(nb.getFunction("output_oxygen_cell"));
        l_output_grid.addAgentFunction(sc.getFunction("output_matrix_grid_cell"));
        // Vasculature
        model.newLayer().addHostFunction(vasculature);
        // Alter
        model.newLayer().addAgentFunction(gc.getFunction("alter"));
        model.newLayer().addHostFunction(alter2);
        // Cell cycle
        auto& l_cycle = model.newLayer();
        l_cycle.addAgentFunction(nb.getFunction("cell_lifecycle"));
        l_cycle.addAgentFunction(sc.getFunction("cell_lifecycle"));
        // Step logging etc (optional)
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
        m_vis.setViewClips(10.0f, 6000.0f);
        auto& nb_agt = m_vis.addAgent("Neuroblastoma");
        auto &sc_agt = m_vis.addAgent("Schwann");
        nb_agt.setXYZVariable("xyz");
        nb_agt.setModel(flamegpu::visualiser::Stock::Models::ICOSPHERE);
        nb_agt.setModelScale(model.getEnvironment().getProperty<float>("R_cell") * 2.0f); // Could improve this in future to use the dynamic rad
        sc_agt.setXYZVariable("xyz");
        sc_agt.setModel(flamegpu::visualiser::Stock::Models::ICOSPHERE);
        nb_agt.setModelScale(model.getEnvironment().getProperty<float>("R_cell") * 2.0f); // Could improve this in future to use the dynamic rad
        // Render the messaging bounding box, -1000 - 1000 each dimension
        {
            auto pen = m_vis.newLineSketch(1, 1, 1, 0.2f);  // white
            // X lines
            pen.addVertex(-2000,  2000,  2000);
            pen.addVertex( 2000,  2000,  2000);
            pen.addVertex(-2000,  2000, -2000);
            pen.addVertex( 2000,  2000, -2000);
            pen.addVertex(-2000, -2000,  2000);
            pen.addVertex( 2000, -2000,  2000);
            pen.addVertex(-2000, -2000, -2000);
            pen.addVertex( 2000, -2000, -2000);
            // Y lines
            pen.addVertex( 2000, -2000,  2000);
            pen.addVertex( 2000,  2000,  2000);
            pen.addVertex( 2000, -2000, -2000);
            pen.addVertex( 2000,  2000, -2000);
            pen.addVertex(-2000, -2000,  2000);
            pen.addVertex(-2000,  2000,  2000);
            pen.addVertex(-2000, -2000, -2000);
            pen.addVertex(-2000,  2000, -2000);
            // Z lines
            pen.addVertex( 2000,  2000, -2000);
            pen.addVertex( 2000,  2000,  2000);
            pen.addVertex( 2000, -2000, -2000);
            pen.addVertex( 2000, -2000,  2000);
            pen.addVertex(-2000,  2000, -2000);
            pen.addVertex(-2000,  2000,  2000);
            pen.addVertex(-2000, -2000, -2000);
            pen.addVertex(-2000, -2000,  2000);
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
