from header import *
import math, sys
import numpy as np

temp_cell_cycle = """
FLAMEGPU_AGENT_FUNCTION(temp_cell_cycle, MsgNone, MsgNone) {
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
    return ALIVE;
}
"""
class logDensity(pyflamegpu.HostFunctionCallback):
    def run(self, FLAMEGPU):
        nb = FLAMEGPU.agent("Neuroblastoma");
        sc = FLAMEGPU.agent("Schwann");
        # Calculate spherical tumour volume
        min_pos_x = min(nb.minFloat("x"), sc.minFloat("x"));
        min_pos_y = min(nb.minFloat("y"), sc.minFloat("y"));
        min_pos_z = min(nb.minFloat("z"), sc.minFloat("z"));
        max_pos_x = max(nb.maxFloat("x"), sc.maxFloat("x"));
        max_pos_y = max(nb.maxFloat("y"), sc.maxFloat("y"));
        max_pos_z = max(nb.maxFloat("z"), sc.maxFloat("z"));
        tumour_volume = 4.0 / 3.0 * math.pi * ((max_pos_x - min_pos_x) / 2.0) * ((max_pos_y - min_pos_y) / 2.0) * ((max_pos_z - min_pos_z) / 2.0);
        # Calculate cells per cubic micron
        tumour_density = ((nb.count() + sc.count()) / tumour_volume) if (tumour_volume >  0) else 0;
        # 1.91e-3 cells per cubic micron (Louis and Shohet, 2015).
        # In a tumour with limited stroma, and whose cells are small, this number is 1e9 cells/cm3 or 1e-3 cells/cubic micron (Del Monte, 2009).
        # However, because cells have different sizes and shapes, the number can be as low as 3.7e7 cells/cm3 or 3.7e-5 cells/cubic micron in a tumour without extracellular structures (Del Monte, 2009).
        #printf("%u: Volume: %g, Cells: %u, Density %g cells/cubic micron %s\n", FLAMEGPU->getStepCounter(), tumour_volume, nb.count() + sc.count(), tumour_density, tumour_density > 1.91e-3 ? " (Too Dense?)" : "");
        # Markdown table log
        if (FLAMEGPU.getStepCounter() == 0):
            # Print header
            print("|Step|Volume|Cells|Density|Note|");
            print("|---|---|---|---|---|");
        print("|%u|%g|%u|%g|%s|"%(FLAMEGPU.getStepCounter(), tumour_volume, nb.count() + sc.count(), tumour_density, ("Too Dense?" if tumour_density > 1.91e-3 else "")));

"""
   Main entrypoint
"""
CELL_COUNT = 1024;
model = pyflamegpu.ModelDescription("PRIMAGE: Neuroblastoma");

defineEnvironment(model, CELL_COUNT);
env = model.Environment();

nb = defineNeuroblastoma(model);
sc = defineSchwann(model);

nb_cc = nb.newRTCFunction("nb_cell_cycle", temp_cell_cycle);
nb_cc.setAgentOutput(nb);
sc_cc = sc.newRTCFunction("sc_cell_cycle", temp_cell_cycle);
sc_cc.setAgentOutput(sc);

forceResolution = defineForceResolution(model);


"""
   Control flow
"""
# Attach init/step/exit functions and exit condition
model.newLayer().addSubModel(forceResolution);
l2 = model.newLayer();
l2.addAgentFunction(nb_cc);
l2.addAgentFunction(sc_cc);
model.addStepFunctionCallback(logDensity().__disown__());


"""
  Create Model Runner
"""
cuda_model = pyflamegpu.CUDASimulation(model, sys.argv);

"""
  Create visualisation
"""
if pyflamegpu.VISUALISATION:
    m_vis = cuda_model.getVisualisation();
    INIT_CAM = 1000 * 1.25;
    # m_vis.setBeginPaused(true);
    m_vis.setInitialCameraLocation(INIT_CAM, INIT_CAM, INIT_CAM);
    m_vis.setCameraSpeed(0.1);
    nb_agt = m_vis.addAgent("Neuroblastoma");
    sc_agt = m_vis.addAgent("Schwann");
    # Position vars are named x, y, z; so they are used by default
    nb_agt.setModel(pyflamegpu.ICOSPHERE);
    nb_agt.setModelScale(22); # 2 * env::R_CELL
    sc_agt.setModel(pyflamegpu.ICOSPHERE);
    sc_agt.setModelScale(22); # 2 * env::R_CELL
    # Render the messaging bounding box, -1000 - 1000 each dimension
    pen = m_vis.newLineSketch(1, 1, 1, 0.2);  # white
    # X lines
    pen.addVertex(-1000,  1000,  1000);
    pen.addVertex( 1000,  1000,  1000);
    pen.addVertex(-1000,  1000, -1000);
    pen.addVertex( 1000,  1000, -1000);
    pen.addVertex(-1000, -1000,  1000);
    pen.addVertex( 1000, -1000,  1000);
    pen.addVertex(-1000, -1000, -1000);
    pen.addVertex( 1000, -1000, -1000);
    # Y lines
    pen.addVertex( 1000, -1000,  1000);
    pen.addVertex( 1000,  1000,  1000);
    pen.addVertex( 1000, -1000, -1000);
    pen.addVertex( 1000,  1000, -1000);
    pen.addVertex(-1000, -1000,  1000);
    pen.addVertex(-1000,  1000,  1000);
    pen.addVertex(-1000, -1000, -1000);
    pen.addVertex(-1000,  1000, -1000);
    # Z lines
    pen.addVertex( 1000,  1000, -1000);
    pen.addVertex( 1000,  1000,  1000);
    pen.addVertex( 1000, -1000, -1000);
    pen.addVertex( 1000, -1000,  1000);
    pen.addVertex(-1000,  1000, -1000);
    pen.addVertex(-1000,  1000,  1000);
    pen.addVertex(-1000, -1000, -1000);
    pen.addVertex(-1000, -1000,  1000);
    m_vis.activate();
    
"""
   Initialisation
"""
if (not cuda_model.getSimulationConfig().input_file):
    t_count_calc = env.getPropertyFloat("rho_tumour") * env.getPropertyFloat("V_tumour") * env.getPropertyFloat("cellularity");
    NB_COUNT = int(math.ceil(t_count_calc * (1 - env.getPropertyFloat("theta_sc"))));
    SC_COUNT = int(math.ceil(t_count_calc * env.getPropertyFloat("theta_sc")));
    # Currently population has not been init, so generate an agent population on the fly
    nb_pop = pyflamegpu.AgentVector(model.Agent("Neuroblastoma"), NB_COUNT);
    sc_pop = pyflamegpu.AgentVector(model.Agent("Schwann"), SC_COUNT);
    for instance in nb_pop:
        u = np.random.uniform() * (env.getPropertyFloat("R_tumour") ** 3.0);
        x1 = np.random.normal();
        x2 = np.random.normal();
        x3 = np.random.normal();

        mag = math.sqrt(x1 * x1 + x2 * x2 + x3 * x3);
        x1 /= mag; x2 /= mag; x3 /= mag;
        c = u**(1./3.);
        instance.setVariableFloat("x", c * x1);
        instance.setVariableFloat("y", c * x2);
        instance.setVariableFloat("z", c * x3);
        instance.setVariableFloat("age", np.random.uniform(0, 24));

    for instance in sc_pop:
        u = np.random.uniform() * (env.getPropertyFloat("R_tumour") ** 3.0);
        x1 = np.random.normal();
        x2 = np.random.normal();
        x3 = np.random.normal();

        mag = math.sqrt(x1 * x1 + x2 * x2 + x3 * x3);
        x1 /= mag; x2 /= mag; x3 /= mag;
        c = u**(1./3.);
        instance.setVariableFloat("x", c * x1);
        instance.setVariableFloat("y", c * x2);
        instance.setVariableFloat("z", c * x3);
        instance.setVariableFloat("age", np.random.uniform(0, 24));

    cuda_model.setPopulationData(nb_pop);
    cuda_model.setPopulationData(sc_pop);


"""
   Execution
"""
cuda_model.simulate();

"""
   Export Pop
"""
#cuda_model.exportData("end.xml");

"""
   Wait for visualisation window to be closed by user after simulation has completed requested steps
"""
if pyflamegpu.VISUALISATION:
    m_vis.join();
