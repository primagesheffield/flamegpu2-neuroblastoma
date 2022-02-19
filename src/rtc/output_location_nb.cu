template<typename M1, typename M2>
__forceinline__ __device__ float calc_R(flamegpu::DeviceAPI<M1, M2>*FLAMEGPU) {
    // The math here isn't as dynamic as in the python model
    // Cycle is tracked differently in each version, which makes tracking the dynamic maths more expensive here
    // Cycle stages haven't changed since the model was first created, so we're probably safe.
    const unsigned int cycle = FLAMEGPU->template getVariable<unsigned int>("cycle");
    const glm::uvec4 cycle_stages = FLAMEGPU->environment.template getProperty<glm::uvec4>("cycle_stages");
    float Ri;
    if (cycle < cycle_stages[0])
        // Ri = <0-1> * 12 / (12 + 4) = <0-0.75>
        Ri = cycle * 0.75f / 12.0f;
    else if (cycle < cycle_stages[1])
        // Ri = 12 / (12 + 4) = 0.75
        Ri = 0.75f;
    else if (cycle < cycle_stages[2])
        // Ri = (12 + (<2-3> - 2) * 4) / (12 + 4) = <12-16> / 16  = <0.75-0.1>
        Ri = 0.75f + (0.25f * (cycle - 18) / (22.0f - 18.0f));
    else
        // Ri = (12 + 4) / (12 + 4) = 1
        Ri = 1.0f;
    return Ri;
}
FLAMEGPU_AGENT_FUNCTION(output_location_nb, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    const glm::vec3 loc = FLAMEGPU->getVariable<glm::vec3>("xyz");
    FLAMEGPU->message_out.setLocation(loc.x, loc.y, loc.z);
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<float>("Rj", calc_R(FLAMEGPU));

    // VALIDATION: Set old location on it 0
    if (FLAMEGPU->getStepCounter() == 0) {
        FLAMEGPU->setVariable<glm::vec3>("old_xyz", loc);
    }
    return flamegpu::ALIVE;
}