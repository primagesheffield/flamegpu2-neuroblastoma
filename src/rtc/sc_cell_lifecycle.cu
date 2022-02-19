#define GRID_MAX_DIMENSIONS 151
#define GMD GRID_MAX_DIMENSIONS
template<typename Mi, typename Mo>
__device__ __forceinline__ glm::ivec3 toGrid(flamegpu::DeviceAPI<Mi, Mo>* FLAMEGPU, const glm::vec3& location) {
    const flamegpu::DeviceEnvironment &env = FLAMEGPU->environment;  // Have to split this out otherwise template keyword required before getProperty
    const float R_voxel = env.getProperty<float>("R_voxel");
    const glm::uvec3 grid_dims = env.getProperty<glm::uvec3>("grid_dims");
    const glm::vec3 span = glm::vec3(grid_dims) * R_voxel * 2.0f;
    const glm::uvec3 grid_origin = env.getProperty<glm::uvec3>("grid_origin");
    return glm::ivec3(
        grid_origin.x + floor((location.x + span.x / 2.0f) / R_voxel / 2.0f),
        grid_origin.y + floor((location.y + span.y / 2.0f) / R_voxel / 2.0f),
        grid_origin.z + floor((location.z + span.z / 2.0f) / R_voxel / 2.0f));
}
/**
 * Notify the NB tracking grid counters of the passed cell's state
 */
template<typename Mi, typename Mo>
__device__ __forceinline__ void increment_grid_nb(flamegpu::DeviceAPI<Mi, Mo>* FLAMEGPU, const glm::ivec3& gid) {
    const flamegpu::DeviceEnvironment& env = FLAMEGPU->environment;  // Have to split this out otherwise template keyword required before getProperty
    // Notify that we are present
    ++env.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nnb_grid")[gid.x][gid.y][gid.z];
    if (FLAMEGPU->template getVariable<int>("apop") == 1) {
        // ++env.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nnba_grid")[gid.x][gid.y][gid.z];
    } else if (FLAMEGPU->template getVariable<int>("necro") == 1) {
        ++env.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nnbn_grid")[gid.x][gid.y][gid.z];
    } else {
        ++env.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nnbl_grid")[gid.x][gid.y][gid.z];
    }
}
/**
 * Notify the SC tracking grid counters of the passed cell's state
 */
template<typename Mi, typename Mo>
__device__ void increment_grid_sc(flamegpu::DeviceAPI<Mi, Mo>* FLAMEGPU, const glm::ivec3& gid) {
    const flamegpu::DeviceEnvironment& env = FLAMEGPU->environment;  // Have to split this out otherwise template keyword required before getProperty
    // Notify that we are present
    ++env.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nsc_grid")[gid.x][gid.y][gid.z];
    if (FLAMEGPU->template getVariable<int>("apop") == 1) {
        // ++env.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nsca_grid")[gid.x][gid.y][gid.z];
    } else if (FLAMEGPU->template getVariable<int>("necro") == 1) {
        ++env.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nscn_grid")[gid.x][gid.y][gid.z];
    } else {
        ++env.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nscl_grid")[gid.x][gid.y][gid.z];
        if (FLAMEGPU->template getVariable<int>("neighbours") < env.getProperty<int>("N_neighbours")) {
            ++env.getMacroProperty<unsigned int, GMD, GMD, GMD>("Nscl_col_grid")[gid.x][gid.y][gid.z];
        }
    }
}
/**
 * Common function used by nb and sc cell lifecycle to shift cells which divide
 */
__device__ __forceinline__ glm::vec3 drift(flamegpu::DeviceAPI<flamegpu::MessageNone, flamegpu::MessageNone>* FLAMEGPU) {
    // Randomly and slightly move a generic cell.
    glm::vec3 location = FLAMEGPU->template getVariable<glm::vec3>("xyz");
    const float R_cell = FLAMEGPU->environment.getProperty<float>("R_cell");
    const glm::vec3 dummy_dir = glm::vec3(
        (FLAMEGPU->random.uniform<float>() * 2) - 1,
        (FLAMEGPU->random.uniform<float>() * 2) - 1,
        (FLAMEGPU->random.uniform<float>() * 2) - 1);
    const glm::vec3 norm_dir = normalize(dummy_dir);
    location.x += 2 * R_cell * norm_dir.x;
    location.y += 2 * R_cell * norm_dir.y;
    location.z += 2 * R_cell * norm_dir.z;
    return location;
}
/**
 * Common function used by nb and sc cell lifecycle to decide whether a cell is dead
 */
__device__ __forceinline__ bool remove(flamegpu::DeviceAPI<flamegpu::MessageNone, flamegpu::MessageNone>* FLAMEGPU) {
    // Remove an apoptotic or necrotic cell if it is engulfed by an immune cell.
    if (FLAMEGPU->getVariable<int>("apop") == 1 || FLAMEGPU->getVariable<int>("necro") == 1) {
        const float P_lysis = FLAMEGPU->environment.getProperty<float>("P_lysis");
        const unsigned int step_size = FLAMEGPU->environment.getProperty<unsigned int>("step_size");
        if (FLAMEGPU->random.uniform<float>() < P_lysis * step_size)
            return true;
    }
    return false;
}
template<typename Mi, typename Mo>
__device__ __forceinline__ bool getChemoState(flamegpu::DeviceAPI<Mi, Mo>* FLAMEGPU) {
    return false;
    // return (FLAMEGPU->getStepCounter() % 504) < 24;
}

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
        // Source 2 : Chemotherapy.
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
        const glm::uvec4 cycle_stages = FLAMEGPU->environment.getProperty<glm::uvec4>("cycle_stages");
        if (getChemoState(FLAMEGPU) && cycle_stages[1] < s_cycle && s_cycle < cycle_stages[2] && FLAMEGPU->random.uniform<float>() < P_apopChemo * step_size) {
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