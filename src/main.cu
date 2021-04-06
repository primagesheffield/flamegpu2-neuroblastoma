#include "header.h"

int main(int argc, const char ** argv) {
    const unsigned int CELL_COUNT = 1024;
    ModelDescription model("PRIMAGE: Neuroblastoma");

    defineEnvironment(model, CELL_COUNT);
    auto& env = model.Environment();

    AgentDescription& nb = defineNeuroblastoma(model);
    AgentDescription& sc = defineSchwann(model);

    SubModelDescription& forceResolution = defineForceResolution(model);


    /**
     * Control flow
     */     
    {   // Attach init/step/exit functions and exit condition
        model.newLayer().addSubModel(forceResolution);
    }

    /**
     * Create Model Runner
     */
    CUDASimulation cuda_model(model, argc, argv);

    /**
     * Create visualisation
     */
#ifdef VISUALISATION
    ModelVis &m_vis = cuda_model.getVisualisation();
    {
        const float INIT_CAM = 1000 * 1.25F;
        m_vis.setInitialCameraLocation(INIT_CAM, INIT_CAM, INIT_CAM);
        m_vis.setCameraSpeed(0.1f);
        auto& nb_agt = m_vis.addAgent("Neuroblastoma");
        auto &sc_agt = m_vis.addAgent("Schwann");
        // Position vars are named x, y, z; so they are used by default
        nb_agt.setModel(Stock::Models::ICOSPHERE);
        nb_agt.setModelScale(22); // 2 * env::R_CELL
        sc_agt.setModel(Stock::Models::ICOSPHERE);
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
            // Y axis
            pen.addVertex( 1000, -1000,  1000);
            pen.addVertex( 1000,  1000,  1000);
            pen.addVertex( 1000, -1000, -1000);
            pen.addVertex( 1000,  1000, -1000);
            pen.addVertex(-1000, -1000,  1000);
            pen.addVertex(-1000,  1000,  1000);
            pen.addVertex(-1000, -1000, -1000);
            pen.addVertex(-1000,  1000, -1000);
            // Z axis
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
        const float t_count_calc = env.getProperty<float>("rho_tumour") * env.getProperty<float>("V_tumour") * env.getProperty<float>("cellularity");
        const unsigned int NB_COUNT = (unsigned int)ceil(t_count_calc * (1 - env.getProperty<float>("theta_sc")));
        const unsigned int SC_COUNT = (unsigned int)ceil(t_count_calc * env.getProperty<float>("theta_sc"));
        // Currently population has not been init, so generate an agent population on the fly
        std::default_random_engine rng;
        std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
        std::normal_distribution<float> normal_dist;
        AgentVector nb_pop(model.Agent("Neuroblastoma"), NB_COUNT);
        AgentVector sc_pop(model.Agent("Schwann"), SC_COUNT);
        for (auto instance: nb_pop) {
            const float u = uniform_dist(rng) * pow(env.getProperty<float>("R_tumour"), 3.0F);
            float x1 = normal_dist(rng);
            float x2 = normal_dist(rng);
            float x3 = normal_dist(rng);

            const float mag = sqrt(x1 * x1 + x2 * x2 + x3 * x3);
            x1 /= mag; x2 /= mag; x3 /= mag;
            const float c = cbrt(u);
            instance.setVariable<float>("x", c * x1);
            instance.setVariable<float>("y", c * x2);
            instance.setVariable<float>("z", c * x3);
        }
        for (auto instance : sc_pop) {
            const float u = uniform_dist(rng) * pow(env.getProperty<float>("R_tumour"), 3.0F);
            float x1 = normal_dist(rng);
            float x2 = normal_dist(rng);
            float x3 = normal_dist(rng);

            const float mag = sqrt(x1 * x1 + x2 * x2 + x3 * x3);
            x1 /= mag; x2 /= mag; x3 /= mag;
            const float c = cbrt(u);
            instance.setVariable<float>("x", c * x1);
            instance.setVariable<float>("y", c * x2);
            instance.setVariable<float>("z", c * x3);
        }
        cuda_model.setPopulationData(nb_pop);
        cuda_model.setPopulationData(sc_pop);
    }

    /**
     * Execution
     */
    cuda_model.simulate();

    /**
     * Export Pop
     */
    cuda_model.exportData("end.xml");

#ifdef VISUALISATION
    m_vis.join();
#endif
    return 0;
}
