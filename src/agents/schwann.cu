#include "header.h"

__device__ __forceinline__ void Schwann_sense(flamegpu::DeviceAPI<flamegpu::MessageNone, flamegpu::MessageNone>* FLAMEGPU) {
    // Let a Schwann cell respond and potentially adapt to the extracellular environment.
    const int s_apop = FLAMEGPU->getVariable<int>("apop");
    const int s_necro = FLAMEGPU->getVariable<int>("necro");
    const unsigned int step_size = FLAMEGPU->environment.getProperty<unsigned int>("step_size");
    if (s_apop == 0 && s_necro == 0) {
        // Detect the presence of stressors.
        // The probability for necrosis due to hypoxia is given in a paper(Warren and Partridge, 2016).
        const float O2 = FLAMEGPU->environment.getProperty<float>("O2");
        const float Cs_O2 = FLAMEGPU->environment.getProperty<float>("Cs_O2");
        const float C50_necro = FLAMEGPU->environment.getProperty<float>("C50_necro");
        const float P_DNA_damagerp = FLAMEGPU->environment.getProperty<float>("P_DNA_damagerp");
        int s_DNA_damage = FLAMEGPU->getVariable<int>("DNA_damage");
        const int s_telo_count = FLAMEGPU->getVariable<int>("telo_count");
        const int s_hypoxia = FLAMEGPU->random.uniform<float>() < (1 - O2 * Cs_O2 / (O2 * Cs_O2 + C50_necro)) ? 1 : 0;
        const int s_nutrient = FLAMEGPU->random.uniform<float>() < (1 - O2 * Cs_O2 / (O2 * Cs_O2 + C50_necro)) ? 0 : 1;

        if (s_DNA_damage == 0) {
            const float P_DNA_damageHypo = FLAMEGPU->environment.getProperty<float>("P_DNA_damageHypo");
            const int telo_critical = FLAMEGPU->environment.getProperty<int>("telo_critical");
            if (FLAMEGPU->random.uniform<float>() < (1 - s_telo_count / static_cast<float>(telo_critical))*step_size) {
                s_DNA_damage = 1;
            } else if (FLAMEGPU->random.uniform<float>() < P_DNA_damageHypo*step_size && s_hypoxia == 1) {
                s_DNA_damage = 1;
            }
        } else if (FLAMEGPU->random.uniform<float>() < P_DNA_damagerp*step_size) {
            s_DNA_damage = 0;
        }

        const int s_DNA_unreplicated = FLAMEGPU->getVariable<int>("DNA_unreplicated");
        if (s_DNA_unreplicated == 1) {
            const float P_unrepDNArp = FLAMEGPU->environment.getProperty<float>("P_unrepDNArp");
            if (FLAMEGPU->random.uniform<float>() < P_unrepDNArp*step_size) {
                FLAMEGPU->setVariable<int>("DNA_unreplicated", 0);
            }
        }
        FLAMEGPU->setVariable<int>("DNA_damage", s_DNA_damage);
        FLAMEGPU->setVariable<int>("hypoxia", s_hypoxia);
        FLAMEGPU->setVariable<int>("nutrient", s_nutrient);

        // Update necrotic signals, starting with the signals from necrotic cells.
        const glm::ivec3 gid = toGrid(FLAMEGPU, FLAMEGPU->getVariable<glm::vec3>("xyz"));
        const glm::uvec3 grid_origin = FLAMEGPU->environment.getProperty<glm::uvec3>("grid_origin");
        const glm::uvec3 grid_dims = FLAMEGPU->environment.getProperty<glm::uvec3>("grid_dims");
        const auto Nnbn = FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nnbn_grid");
        const auto Nscn = FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nscn_grid");
        unsigned int dummy_Nn = Nnbn[gid.x][gid.y][gid.z] + Nscn[gid.x][gid.y][gid.z];
        if (gid.x > grid_origin.x)
            dummy_Nn += Nnbn[gid.x - 1][gid.y][gid.z] + Nscn[gid.x - 1][gid.y][gid.z];
        if (gid.x + 1 < grid_origin.x + grid_dims.x)
            dummy_Nn += Nnbn[gid.x + 1][gid.y][gid.z] + Nscn[gid.x + 1][gid.y][gid.z];
        if (gid.y > grid_origin.y)
            dummy_Nn += Nnbn[gid.x][gid.y - 1][gid.z] + Nscn[gid.x][gid.y - 1][gid.z];
        if (gid.y + 1 < grid_origin.y + grid_dims.y)
            dummy_Nn += Nnbn[gid.x][gid.y + 1][gid.z] + Nscn[gid.x][gid.y + 1][gid.z];
        if (gid.z > grid_origin.z)
            dummy_Nn += Nnbn[gid.x][gid.y][gid.z - 1] + Nscn[gid.x][gid.y][gid.z - 1];
        if (gid.z + 1 < grid_origin.z + grid_dims.z)
            dummy_Nn += Nnbn[gid.x][gid.y][gid.z + 1] + Nscn[gid.x][gid.y][gid.z + 1];


        const float P_necroIS = FLAMEGPU->environment.getProperty<float>("P_necroIS");
        int s_necro_signal = FLAMEGPU->getVariable<int>("necro_signal");
        int stress = 0;

        for (int j = 0; j < dummy_Nn; ++j) {
            if (FLAMEGPU->random.uniform<float>() < P_necroIS * step_size) {
                s_necro_signal += 1 * step_size;
                stress = 1;
            }
        }

        // The contribution of glycolysis to necrosis is time-independent.
        const float glycoEff = FLAMEGPU->environment.getProperty<float>("glycoEff");
        const float P_necrorp = FLAMEGPU->environment.getProperty<float>("P_necrorp");
        if (s_nutrient == 1) {
            if (s_hypoxia == 0) {
                FLAMEGPU->setVariable<int>("ATP", 1);
                if (s_necro_signal > 0 && (FLAMEGPU->random.uniform<float>() < P_necrorp * step_size) && stress == 0)
                    s_necro_signal -= 1 * step_size;
            } else if (FLAMEGPU->random.uniform<float>() < glycoEff) {
                FLAMEGPU->setVariable<int>("ATP", 1);
                s_necro_signal += 1 * step_size;
            } else {
                FLAMEGPU->setVariable<int>("ATP", 0);
                s_necro_signal += 2 * step_size;
            }
        } else {
            FLAMEGPU->setVariable<int>("ATP", 0);
            s_necro_signal += 1 * step_size;
        }
        FLAMEGPU->setVariable<int>("necro_signal", s_necro_signal);

        // Update apoptotic signals.
        // Source 1: CAS is activated by DNA damage or hypoxia.
        // Source 2 : Chemotherapy, assuming drug delivery is instantaneous.
        const float P_apopChemo = FLAMEGPU->environment.getProperty<float>("P_apopChemo");
        const float P_apoprp = FLAMEGPU->environment.getProperty<float>("P_apoprp");
        int s_apop_signal = FLAMEGPU->getVariable<int>("apop_signal");
        const unsigned int s_cycle = FLAMEGPU->getVariable<unsigned int>("cycle");
        const int s_ATP = FLAMEGPU->getVariable<int>("ATP");  // Could set this above, rather than get
        stress = 0;
        if ((s_DNA_damage == 1 || s_hypoxia == 1) && s_ATP == 1) {
            s_apop_signal += 1 * step_size;
            stress = 1;
        }
	const float chemo0 = FLAMEGPU.environment.getProperty<float, 6>("chemo_effects", 0);
        const float chemo1 = FLAMEGPU.environment.getProperty<float, 6>("chemo_effects", 1);
        const float chemo2 = FLAMEGPU.environment.getProperty<float, 6>("chemo_effects", 2);
        const float chemo3 = FLAMEGPU.environment.getProperty<float, 6>("chemo_effects", 3);
        const float chemo4 = FLAMEGPU.environment.getProperty<float, 6>("chemo_effects", 4);
        const float chemo5 = FLAMEGPU.environment.getProperty<float, 6>("chemo_effects", 5);
        const glm::uvec4 cycle_stages = FLAMEGPU->environment.getProperty<glm::uvec4>("cycle_stages");
	int chemo = 0;
	if(CHEMO_ACTIVE && FLAMEGPU->random.uniform<float>()<(chemo0+chemo1+chemo2+chemo3+chemo4+chemo5)/6)
                {
                        chemo = 1;
                        break;
                }
        if (chemo == 1 && cycle_stages[1] < s_cycle && s_cycle < cycle_stages[2] && FLAMEGPU->random.uniform<float>() < P_apopChemo * step_size) {
            s_apop_signal += 1 * step_size;
            stress = 1;
        }
        if (s_apop_signal > 0 && (FLAMEGPU->random.uniform<float>() < P_apoprp*step_size) && stress == 0) {
            s_apop_signal -= 1 * step_size;
        }
        FLAMEGPU->setVariable<int>("apop_signal", s_apop_signal);

        // Update apoptotic status and necrotic status.
        const int apop_critical = FLAMEGPU->environment.getProperty<int>("apop_critical");
        const int s_necro_critical = FLAMEGPU->getVariable<int>("necro_critical");
        if (s_apop_signal >= apop_critical) {
            FLAMEGPU->setVariable<int>("mobile", 0);
            FLAMEGPU->setVariable<int>("ATP", 0);
            FLAMEGPU->setVariable<int>("apop", 1);
        } else if (s_necro_signal >= s_necro_critical) {
            FLAMEGPU->setVariable<int>("mobile", 0);
            FLAMEGPU->setVariable<int>("ATP", 0);
            FLAMEGPU->setVariable<int>("necro", 1);
        }

    } else if (s_apop == 1 && s_necro == 0) {
        // Secondary necrosis.
        const float P_2ndnecro = FLAMEGPU->environment.getProperty<float>("P_2ndnecro");
        if (FLAMEGPU->random.uniform<float>() < P_2ndnecro * step_size) {
            FLAMEGPU->setVariable<int>("apop", 0);
            FLAMEGPU->setVariable<int>("necro", 1);
        }
    }
}
__device__ __forceinline__ void Schwann_cell_cycle(flamegpu::DeviceAPI<flamegpu::MessageNone, flamegpu::MessageNone>* FLAMEGPU) {
    // Progress through the Schwann cell's cell cycle.
    // In the cell cycle, 0 = G0, 1 = G1 / S, 2 = S / G2, 3 = G2 / M, 4 = division.
    // Regulatory mechanisms :
    //      1. Contact inhibition.
    //      2. ATP availability.
    //      3. Apoptotic / Necrotic status.
    //      4. Extent of differentiation.
    //      5. Stimulation from neuroblasts.
    //      6. DNA damage and unreplicated DNA.
    //      7. Hypoxia.
    // DNA damage and hypoxia must both be off during G1 and S because each can switch on p21 and p27 to arrest cycling.
    // Either DNA damage or unreplicated DNA can switch off CDC25C to arrest G2 / M transition.

    // Decide our voxel
    const glm::ivec3 gid = toGrid(FLAMEGPU, FLAMEGPU->getVariable<glm::vec3>("xyz"));
    // Count the neuroblasts and Schwann cells in the 3D von Neumann neighbourhood.
    const glm::uvec3 grid_origin = FLAMEGPU->environment.getProperty<glm::uvec3>("grid_origin");
    const glm::uvec3 grid_dims = FLAMEGPU->environment.getProperty<glm::uvec3>("grid_dims");
    const auto Nnbl = FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nnbl_grid");
    const auto Nscl = FLAMEGPU->environment.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nscl_grid");
    unsigned int dummy_Nnbl = Nnbl[gid.x][gid.y][gid.z];
    unsigned int dummy_Nscl = Nscl[gid.x][gid.y][gid.z];
    if (gid.x > grid_origin.x) {
        dummy_Nnbl += Nnbl[gid.x - 1][gid.y][gid.z];
        dummy_Nscl += Nscl[gid.x - 1][gid.y][gid.z];
    }
    if (gid.x + 1 < grid_origin.x + grid_dims.x) {
        dummy_Nnbl += Nnbl[gid.x + 1][gid.y][gid.z];
        dummy_Nscl += Nscl[gid.x + 1][gid.y][gid.z];
    }
    if (gid.y > grid_origin.y) {
        dummy_Nnbl += Nnbl[gid.x][gid.y - 1][gid.z];
        dummy_Nscl += Nscl[gid.x][gid.y - 1][gid.z];
    }
    if (gid.y + 1 < grid_origin.y + grid_dims.y) {
        dummy_Nnbl += Nnbl[gid.x][gid.y + 1][gid.z];
        dummy_Nscl += Nscl[gid.x][gid.y + 1][gid.z];
    }
    if (gid.z > grid_origin.z) {
        dummy_Nnbl += Nnbl[gid.x][gid.y][gid.z - 1];
        dummy_Nscl += Nscl[gid.x][gid.y][gid.z - 1];
    }
    if (gid.z + 1 < grid_origin.z + grid_dims.z) {
        dummy_Nnbl += Nnbl[gid.x][gid.y][gid.z + 1];
        dummy_Nscl += Nscl[gid.x][gid.y][gid.z + 1];
    }

    const glm::uvec4 cycle_stages = FLAMEGPU->environment.getProperty<glm::uvec4>("cycle_stages");
    const int N_neighbours = FLAMEGPU->environment.getProperty<int>("N_neighbours");
    const unsigned int step_size = FLAMEGPU->environment.getProperty<unsigned int>("step_size");
    const float scpro_jux = FLAMEGPU->environment.getProperty<float>("scpro_jux");
    const float scpro_para = FLAMEGPU->environment.getProperty<float>("scpro_para");
    const unsigned int Nnbl_count = FLAMEGPU->environment.getProperty<unsigned int>("Nnbl_count");
    const unsigned int Nscl_count = FLAMEGPU->environment.getProperty<unsigned int>("Nscl_count");

    // The influence of neuroblasts on Schwann cell proliferation.
    bool dummy_scpro;
    if (FLAMEGPU->random.uniform<float>() < step_size*scpro_jux*dummy_Nnbl / static_cast<float>(dummy_Nnbl + dummy_Nscl)) {
        dummy_scpro = true;
    } else if (FLAMEGPU->random.uniform<float>() < step_size*scpro_para*Nnbl_count / static_cast<float>(Nnbl_count + Nscl_count)) {
        dummy_scpro = true;
    } else {
        dummy_scpro = false;
    }
    const float P_cycle_sc = FLAMEGPU->environment.getProperty<float>("P_cycle_sc");
    const bool dummy_scycle = (dummy_scpro == 1 || FLAMEGPU->random.uniform<float>() < P_cycle_sc) ? true : false;

    // In the cell cycle, 0[12] = G0, 1[6] = G1/S, 2[4] = S/G2, 3[2] = G2/M, 4[0] = division.
    // In the cell cycle, 0-11 = G0, 12-17 = G1/S, 18-21 = S/G2, 22-23 = G2/M, 24+ = division.
    unsigned int s_cycle = FLAMEGPU->getVariable<unsigned int>("cycle");
    const int s_neighbours = FLAMEGPU->getVariable<int>("neighbours");
    const int s_ATP = FLAMEGPU->getVariable<int>("ATP");
    const int s_apop = FLAMEGPU->getVariable<int>("apop");
    const int s_necro = FLAMEGPU->getVariable<int>("necro");
    const int s_DNA_damage = FLAMEGPU->getVariable<int>("DNA_damage");
    const int s_DNA_unreplicated = FLAMEGPU->getVariable<int>("DNA_unreplicated");
    const int s_hypoxia = FLAMEGPU->getVariable<int>("hypoxia");
    if (dummy_scycle && s_neighbours <= N_neighbours && s_ATP == 1 && s_apop == 0 && s_necro == 0) {
        if (s_cycle < cycle_stages[0]) {
            if (s_cycle == 0) {
                if (s_DNA_damage == 0 && s_hypoxia == 0) {
                    s_cycle += step_size;
                }
            } else if (s_DNA_damage == 0 && s_hypoxia == 0) {
                s_cycle += step_size;
            }
        } else if (s_cycle < cycle_stages[1]) {
            if (s_DNA_damage == 0 && s_hypoxia == 0) {
                s_cycle += step_size;
                if (s_DNA_unreplicated == 0) {
                    const float P_unrepDNA = FLAMEGPU->environment.getProperty<float>("P_unrepDNA");
                    const float P_unrepDNAHypo = FLAMEGPU->environment.getProperty<float>("P_unrepDNAHypo");
                    if (FLAMEGPU->random.uniform<float>() < P_unrepDNA * step_size) {
                        FLAMEGPU->setVariable<int>("DNA_unreplicated", 1);
                    } else if (FLAMEGPU->random.uniform<float>() < P_unrepDNAHypo * step_size && s_hypoxia == 1) {
                        FLAMEGPU->setVariable<int>("DNA_unreplicated", 1);
                    }
                }
            }
        } else if (s_cycle < cycle_stages[2]) {
            s_cycle += step_size;
            if (s_cycle >= cycle_stages[2] && (s_DNA_damage == 1 || s_DNA_unreplicated == 1)) {
                s_cycle -= step_size;
            }
        } else if (s_cycle < cycle_stages[3]) {
            s_cycle += step_size;
        }
    }
    FLAMEGPU->setVariable<unsigned int>("cycle", s_cycle);
}
__device__ __forceinline__ bool Schwann_divide(flamegpu::DeviceAPI<flamegpu::MessageNone, flamegpu::MessageNone>* FLAMEGPU) {
    const int s_apop = FLAMEGPU->getVariable<int>("apop");
    const int s_necro = FLAMEGPU->getVariable<int>("necro");
    if (!(s_apop == 0 && s_necro == 0))
        return false;
    // Division of a living Schwann cell.
    // At the end of the cell cycle, the cell divides.
    //    (a)Move it back to the beginning of the cycle.
    //    (b)If it has at least one telomere unit, shorten its telomere.
    const unsigned int s_cycle = FLAMEGPU->getVariable<unsigned int>("cycle");
    const int s_telo_count = FLAMEGPU->getVariable<int>("telo_count");
    const glm::uvec4 cycle_stages = FLAMEGPU->environment.getProperty<glm::uvec4>("cycle_stages");
    if (s_cycle >= cycle_stages[3]) {
        FLAMEGPU->setVariable<unsigned int>("cycle", 0);
        if (s_telo_count > 0)
            FLAMEGPU->setVariable<int>("telo_count", s_telo_count - 1);
        return true;
    }
    return false;
}


FLAMEGPU_AGENT_FUNCTION(sc_cell_lifecycle, flamegpu::MessageNone, flamegpu::MessageNone) {
    Schwann_sense(FLAMEGPU);
    Schwann_cell_cycle(FLAMEGPU);

    if (remove(FLAMEGPU)) {
        return flamegpu::DEAD;  // Kill cell
    } else if (Schwann_divide(FLAMEGPU)) {
        const glm::vec3 newLoc = drift(FLAMEGPU);
        // Spatial coordinates (integration with imaging biomarkers).
        FLAMEGPU->agent_out.setVariable<glm::vec3>("xyz", newLoc);
        // Initial conditions
        FLAMEGPU->agent_out.setVariable<glm::vec3>("Fxyz", FLAMEGPU->getVariable<glm::vec3>("Fxyz"));  // This could be left to default init?
        FLAMEGPU->agent_out.setVariable<float>("overlap", FLAMEGPU->getVariable<float>("overlap"));  // This could be left to default init?
        FLAMEGPU->agent_out.setVariable<int>("neighbours", 0);  // This could be left to default init?
        FLAMEGPU->agent_out.setVariable<int>("mobile", FLAMEGPU->getVariable<int>("mobile"));
        FLAMEGPU->agent_out.setVariable<int>("ATP", FLAMEGPU->getVariable<int>("ATP"));
        FLAMEGPU->agent_out.setVariable<unsigned int>("cycle", FLAMEGPU->getVariable<unsigned int>("cycle"));
        FLAMEGPU->agent_out.setVariable<int>("apop", FLAMEGPU->getVariable<int>("apop"));
        FLAMEGPU->agent_out.setVariable<int>("apop_signal", FLAMEGPU->getVariable<int>("apop_signal"));
        FLAMEGPU->agent_out.setVariable<int>("necro", FLAMEGPU->getVariable<int>("necro"));
        FLAMEGPU->agent_out.setVariable<int>("necro_signal", FLAMEGPU->getVariable<int>("necro_signal"));
        FLAMEGPU->agent_out.setVariable<int>("necro_critical", FLAMEGPU->getVariable<int>("necro_critical"));
        FLAMEGPU->agent_out.setVariable<int>("telo_count", FLAMEGPU->getVariable<int>("telo_count"));
        FLAMEGPU->agent_out.setVariable<float>("degdiff", FLAMEGPU->getVariable<float>("degdiff"));
        FLAMEGPU->agent_out.setVariable<float>("cycdiff", FLAMEGPU->getVariable<float>("cycdiff"));
        // Attribute Layer 1.
        FLAMEGPU->agent_out.setVariable<int>("hypoxia", FLAMEGPU->getVariable<int>("hypoxia"));
        FLAMEGPU->agent_out.setVariable<int>("nutrient", FLAMEGPU->getVariable<int>("nutrient"));
        FLAMEGPU->agent_out.setVariable<int>("DNA_damage", FLAMEGPU->getVariable<int>("DNA_damage"));
        FLAMEGPU->agent_out.setVariable<int>("DNA_unreplicated", FLAMEGPU->getVariable<int>("DNA_unreplicated"));
        // Internal
        FLAMEGPU->agent_out.setVariable<float>("force_magnitude", 0);  // This could be left to default init?
    }
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(output_matrix_grid_cell, flamegpu::MessageNone, flamegpu::MessageNone) {
    const glm::ivec3 gid = toGrid(FLAMEGPU, FLAMEGPU->getVariable<glm::vec3>("xyz"));
    increment_grid_sc(FLAMEGPU, gid);
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(sc_validation, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("apop") == 0 && FLAMEGPU->getVariable<int>("necro") == 0) {
        ++FLAMEGPU->environment.getMacroProperty<unsigned int>("validation_Nscl");
    }
    return flamegpu::ALIVE;
}

flamegpu::AgentDescription &defineSchwann(flamegpu::ModelDescription& model) {
    auto& sc = model.newAgent("Schwann");
    // Data Layer 0 (integration with imaging biomarkers)
    {
        sc.newVariable<float, 3>("xyz");
    }
    // Initial Conditions.
    {
        sc.newVariable<float, 3>("Fxyz");
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
    // Agent functions
    {
        sc.newFunction("output_matrix_grid_cell", output_matrix_grid_cell);
        auto &t = sc.newFunction("sc_cell_lifecycle", sc_cell_lifecycle);
        t.setAllowAgentDeath(true);
        t.setAgentOutput(sc);
        sc.newFunction("sc_validation", sc_validation);
    }
    return sc;
}

void initSchwann(flamegpu::HostAPI &FLAMEGPU) {
    auto SC =  FLAMEGPU.agent("Schwann");
    if (FLAMEGPU.agent("Schwann").count() != 0)
        return;  // SC agents must have been loaded already

    // Env properties required for initialising NB agents
    const float R_tumour = FLAMEGPU.environment.getProperty<float>("R_tumour");
    const int cycle_sc = FLAMEGPU.environment.getProperty<int>("cycle_sc");
    const std::array<unsigned int, 4> cycle_stages = FLAMEGPU.environment.getProperty<unsigned int, 4>("cycle_stages");
    const int apop_sc = FLAMEGPU.environment.getProperty<int>("apop_sc");
    const int apop_signal_sc = FLAMEGPU.environment.getProperty<int>("apop_signal_sc");
    const int necro_sc = FLAMEGPU.environment.getProperty<int>("necro_sc");
    const int necro_signal_sc = FLAMEGPU.environment.getProperty<int>("necro_signal_sc");
    const int telo_count_sc = FLAMEGPU.environment.getProperty<int>("telo_count_sc");

    // Env properties required for calculating agent count
    const float rho_tumour = FLAMEGPU.environment.getProperty<float>("rho_tumour");
    const float V_tumour = FLAMEGPU.environment.getProperty<float>("V_tumour");
    const float cellularity = FLAMEGPU.environment.getProperty<float>("cellularity");
    const float theta_sc = FLAMEGPU.environment.getProperty<float>("theta_sc");

    const unsigned int SC_COUNT = (unsigned int)ceil(rho_tumour * V_tumour * cellularity * theta_sc);
    unsigned int validation_Nscl = 0;
    for (unsigned int i = 0; i < SC_COUNT; ++i) {
        auto agt = SC.newAgent();
        // Data Layer 0 (integration with imaging biomarkers).
        agt.setVariable<glm::vec3>("xyz",
            glm::vec3(-R_tumour + (FLAMEGPU.random.uniform<float>() * 2 * R_tumour),
                -R_tumour + (FLAMEGPU.random.uniform<float>() * 2 * R_tumour),
                -R_tumour + (FLAMEGPU.random.uniform<float>() * 2 * R_tumour)));
        // Initial conditions.
        agt.setVariable<glm::vec3>("Fxyz", glm::vec3(0));
        agt.setVariable<float>("overlap", 0);
        agt.setVariable<int>("neighbours", 0);
        agt.setVariable<int>("mobile", 1);
        agt.setVariable<int>("ATP", 1);
        if (cycle_sc >= 0) {
            agt.setVariable<unsigned int>("cycle", static_cast<unsigned int>(cycle_sc));
        } else {
            // Weird init, because Py model has uniform chance per stage
            // uniform chance within stage
            const int stage = FLAMEGPU.random.uniform<int>(0, 3);  // Random int in range [0, 3]
            const unsigned int stage_start = stage == 0 ? 0 : cycle_stages[stage - 1];
            const unsigned int stage_extra = static_cast<unsigned int>(FLAMEGPU.random.uniform<float>() * static_cast<float>(cycle_stages[stage] - stage_start));
            agt.setVariable<unsigned int>("cycle", stage_start + stage_extra);
        }
        agt.setVariable<int>("apop", apop_sc < 0 ? 0 : apop_sc);
        agt.setVariable<int>("apop_signal", apop_signal_sc < 0 ? 0 : apop_signal_sc);
        agt.setVariable<int>("necro", necro_sc < 0 ? 0 : necro_sc);
        agt.setVariable<int>("necro_signal", necro_signal_sc < 0 ? 0 : necro_signal_sc);
        validation_Nscl += (apop_sc < 0 ? 0 : apop_sc) == 0 && (necro_sc < 0 ? 0 : necro_sc) == 0 ? 1 : 0;
        agt.setVariable<int>("necro_critical", FLAMEGPU.random.uniform<int>(3, 168));  // Random int in range [3, 168]
        agt.setVariable<int>("telo_count", telo_count_sc < 0 ? FLAMEGPU.random.uniform<int>(25, 35) : telo_count_sc);  // Random int in range [25, 35]
        // Attribute Layer 1
        agt.setVariable<int>("hypoxia", 0);
        agt.setVariable<int>("nutrient", 1);
        agt.setVariable<int>("DNA_damage", 0);
        agt.setVariable<int>("DNA_unreplicated", 0);
    }
    FLAMEGPU.environment.setProperty<unsigned int>("validation_Nscl", validation_Nscl);
}
