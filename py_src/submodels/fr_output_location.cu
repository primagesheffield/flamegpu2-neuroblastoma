// Used by both Neuroblastoma and Schwann
FLAMEGPU_AGENT_FUNCTION(output_location, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    FLAMEGPU->message_out.setLocation(
        FLAMEGPU->getVariable<float>("x"),
        FLAMEGPU->getVariable<float>("y"),
        FLAMEGPU->getVariable<float>("z"));
    FLAMEGPU->message_out.setVariable<unsigned int>("id", 0);  // Currently unused, FGPU2 will have auto agent ids in future
    return flamegpu::ALIVE;
}