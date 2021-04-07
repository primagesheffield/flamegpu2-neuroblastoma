from pyflamegpu import *

def defineSchwann(model):
    sc = model.newAgent("Schwann");
    sc.newVariableFloat("x");
    sc.newVariableFloat("y");
    sc.newVariableFloat("z");
    # Temporary replacement for cell cycle
    sc.newVariableFloat("age");
    return sc;
