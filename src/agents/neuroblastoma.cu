#include "header.h"

AgentDescription& defineNeuroblastoma(ModelDescription& model) {
    auto& nb = model.newAgent("Neuroblastoma");
    nb.newVariable<float>("x");
    nb.newVariable<float>("y");
    nb.newVariable<float>("z");
    return nb;
}
