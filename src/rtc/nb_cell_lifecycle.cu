#define GRID_MAX_DIMENSIONS 151
#define GMD GRID_MAX_DIMENSIONS
/**
 * Converts a continuous location to a discrete grid position
 */
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

__device__ __forceinline__ void Neuroblastoma_sense(flamegpu::DeviceAPI<flamegpu::MessageNone, flamegpu::MessageNone> *FLAMEGPU) {
    // Let a neuroblastoma cell respond and potentially adapt to the extracellular environment.
    const int s_apop = FLAMEGPU->getVariable<int>("apop");
    const int s_necro = FLAMEGPU->getVariable<int>("necro");
    const unsigned int step_size = FLAMEGPU->environment.getProperty<unsigned int>("step_size");
    if (s_apop == 0 && s_necro == 0) {
        // Update attribute layer 1, part 1.
        // Detect the presence of stressors.
        // The probability for necrosis due to hypoxia is given in a paper (Warren and Partridge, 2016).
        // Note that the effects of p53 and p73 on DNA_damage are considered before p53 and p73 are updated.
        const float O2 = FLAMEGPU->environment.getProperty<float>("O2");
        const float Cs_O2 = FLAMEGPU->environment.getProperty<float>("Cs_O2");
        const float C50_necro = FLAMEGPU->environment.getProperty<float>("C50_necro");
        int s_DNA_damage = FLAMEGPU->getVariable<int>("DNA_damage");
        const int s_hypoxia = FLAMEGPU->random.uniform<float>() < (1 - O2 * Cs_O2 / (O2 * Cs_O2 + C50_necro)) ? 1 : 0;
        const int s_nutrient = FLAMEGPU->random.uniform<float>() < (1 - O2 * Cs_O2 / (O2 * Cs_O2 + C50_necro)) ? 0 : 1;

        if (s_DNA_damage == 1) {
            if (FLAMEGPU->getVariable<int>("p53") == 1 || FLAMEGPU->getVariable<int>("p73") == 1)
                s_DNA_damage = 0;
        } else {
            const float P_DNA_damageHypo = FLAMEGPU->environment.getProperty<float>("P_DNA_damageHypo");
            const int telo_critical = FLAMEGPU->environment.getProperty<int>("telo_critical");
            if ((FLAMEGPU->random.uniform<float>() < (1.0f - static_cast<float>(FLAMEGPU->getVariable<int>("telo_count")) / telo_critical) * step_size)
            || (FLAMEGPU->random.uniform<float>() < P_DNA_damageHypo * step_size && s_hypoxia == 1)) {
                s_DNA_damage = 1;
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
            dummy_Nn += Nnbn[gid.x][gid.y - 1][gid.z] +  Nscn[gid.x][gid.y - 1][gid.z];
        if (gid.y + 1 < grid_origin.y + grid_dims.y)
            dummy_Nn += Nnbn[gid.x][gid.y + 1][gid.z] +  Nscn[gid.x][gid.y + 1][gid.z];
        if (gid.z > grid_origin.z)
            dummy_Nn += Nnbn[gid.x][gid.y][gid.z - 1] +  Nscn[gid.x][gid.y][gid.z - 1];
        if (gid.z + 1 < grid_origin.z + grid_dims.z)
            dummy_Nn += Nnbn[gid.x][gid.y][gid.z + 1] +  Nscn[gid.x][gid.y][gid.z + 1];

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

        // Update attribute layer 3,part 1.
        // Let the intracellular signalling molecules respond to changes.
        // Note that the effects of p53 and p73 on HIF are considered before p53 and p73 are updated.
        const float s_MYCN_fn = FLAMEGPU->getVariable<float>("MYCN_fn");
        const float s_MAPK_RAS_fn = FLAMEGPU->getVariable<float>("MAPK_RAS_fn");
        const float s_JAB1_fn = FLAMEGPU->getVariable<float>("JAB1_fn");
        const float s_CHK1_fn = FLAMEGPU->getVariable<float>("CHK1_fn");
        const float s_ID2_fn = FLAMEGPU->getVariable<float>("ID2_fn");
        const float s_IAP2_fn = FLAMEGPU->getVariable<float>("IAP2_fn");
        const float s_HIF_fn = FLAMEGPU->getVariable<float>("HIF_fn");
        const float s_BNIP3_fn = FLAMEGPU->getVariable<float>("BNIP3_fn");
        const float s_VEGF_fn = FLAMEGPU->getVariable<float>("VEGF_fn");
        const float s_p53_fn = FLAMEGPU->getVariable<float>("p53_fn");
        const float s_p73_fn = FLAMEGPU->getVariable<float>("p73_fn");
        const float s_p21_fn = FLAMEGPU->getVariable<float>("p21_fn");
        const float s_p27_fn = FLAMEGPU->getVariable<float>("p27_fn");
        const float s_Bcl2_Bclxl_fn = FLAMEGPU->getVariable<float>("Bcl2_Bclxl_fn");
        const float s_BAK_BAX_fn = FLAMEGPU->getVariable<float>("BAK_BAX_fn");
        const float s_CAS_fn = FLAMEGPU->getVariable<float>("CAS_fn");

        const int s_MYCN = FLAMEGPU->random.uniform<float>() < s_MYCN_fn ? 1 : 0;
        const int s_MAPK_RAS = FLAMEGPU->random.uniform<float>() < s_MAPK_RAS_fn ? 1 : 0;
        const int s_JAB1 = FLAMEGPU->random.uniform<float>() < s_JAB1_fn ? 1 : 0;
        const int s_CHK1 = (FLAMEGPU->random.uniform<float>() < s_CHK1_fn && s_DNA_damage == 1) || (FLAMEGPU->random.uniform<float>() < s_CHK1_fn && s_DNA_damage == 1 && s_MYCN == 1) ? 1 : 0;
        const int s_ID2 = FLAMEGPU->random.uniform<float>() < s_ID2_fn && s_MYCN == 1 ? 1 : 0;
        const int s_IAP2 = FLAMEGPU->random.uniform<float>() < s_IAP2_fn && s_hypoxia == 1 ? 1 : 0;
        int s_p53 = FLAMEGPU->getVariable<int>("p53");
        int s_p73 = FLAMEGPU->getVariable<int>("p73");
        const int s_HIF = ((FLAMEGPU->random.uniform<float>() < s_HIF_fn && s_hypoxia == 1) || (FLAMEGPU->random.uniform<float>() < s_HIF_fn && s_hypoxia == 1 && s_JAB1 == 1)) && !(s_p53 == 1 || s_p73 == 1) ? 1 : 0;
        const int s_BNIP3 = FLAMEGPU->random.uniform<float>() < s_BNIP3_fn && s_HIF == 1 ? 1 : 0;
        const int s_VEGF = FLAMEGPU->random.uniform<float>() < s_VEGF_fn && s_HIF == 1 ? 1 : 0;
        const int s_MYCN_amp = FLAMEGPU->getVariable<int>("MYCN_amp");
        s_p53 = ((FLAMEGPU->random.uniform<float>() < s_p53_fn && s_DNA_damage == 1) ||
                (FLAMEGPU->random.uniform<float>() < s_p53_fn && s_DNA_damage == 1 && s_MYCN == 1) ||
                (FLAMEGPU->random.uniform<float>() < s_p53_fn && s_HIF == 1) ||
                (FLAMEGPU->random.uniform<float>() < s_p53_fn&& s_HIF == 1 && s_MYCN == 1)) &&
                !(s_MYCN == 1 && s_MYCN_amp == 1)
                ? 1 : 0;
        s_p73 = (FLAMEGPU->random.uniform<float>() < s_p73_fn && s_CHK1 == 1) ||
                (FLAMEGPU->random.uniform<float>() < s_p73_fn && s_HIF == 1)
                ? 1 : 0;
        const int s_p21 = ((FLAMEGPU->random.uniform<float>() < s_p21_fn && s_HIF == 1) || (FLAMEGPU->random.uniform<float>() < s_p21_fn && s_p53 == 1)) && !(s_MAPK_RAS == 1 || s_MYCN == 1) ? 1 : 0;
        const int s_p27 = ((FLAMEGPU->random.uniform<float>() < s_p27_fn && s_HIF == 1) || (FLAMEGPU->random.uniform<float>() < s_p27_fn && s_p53 == 1)) && !(s_MAPK_RAS == 1 || s_MYCN == 1) ? 1 : 0;
        const int s_Bcl2_Bclxl = (FLAMEGPU->random.uniform<float>() < s_Bcl2_Bclxl_fn && !(s_BNIP3 == 1 || s_p53 == 1 || s_p73 == 1)) ? 1 : 0;
        const int s_BAK_BAX = ((FLAMEGPU->random.uniform<float>() < s_BAK_BAX_fn && s_hypoxia == 1) || (FLAMEGPU->random.uniform<float>() < s_BAK_BAX_fn && s_p53 == 1) || (FLAMEGPU->random.uniform<float>() < s_BAK_BAX_fn && s_p73 == 1)) && !(s_Bcl2_Bclxl == 1 || s_IAP2 == 1) ? 1 : 0;
        const int s_ATP = FLAMEGPU->getVariable<int>("ATP");
        const int s_CAS = ((FLAMEGPU->random.uniform<float>() < s_CAS_fn && s_BAK_BAX == 1) || (FLAMEGPU->random.uniform<float>() < s_CAS_fn && s_hypoxia == 1)) && s_ATP == 1 ? 1 : 0;

        FLAMEGPU->setVariable<int>("MYCN", s_MYCN);
        FLAMEGPU->setVariable<int>("MAPK_RAS", s_MAPK_RAS);
        FLAMEGPU->setVariable<int>("JAB1", s_JAB1);
        FLAMEGPU->setVariable<int>("CHK1", s_CHK1);
        FLAMEGPU->setVariable<int>("ID2", s_ID2);
        FLAMEGPU->setVariable<int>("IAP2", s_IAP2);
        FLAMEGPU->setVariable<int>("HIF", s_HIF);
        FLAMEGPU->setVariable<int>("BNIP3", s_BNIP3);
        FLAMEGPU->setVariable<int>("VEGF", s_VEGF);
        FLAMEGPU->setVariable<int>("p53", s_p53);
        FLAMEGPU->setVariable<int>("p73", s_p73);
        FLAMEGPU->setVariable<int>("p21", s_p21);
        FLAMEGPU->setVariable<int>("p27", s_p27);
        FLAMEGPU->setVariable<int>("Bcl2_Bclxl", s_Bcl2_Bclxl);
        FLAMEGPU->setVariable<int>("BAK_BAX", s_BAK_BAX);
        FLAMEGPU->setVariable<int>("CAS", s_CAS);

        // Update attribute layer 1, part 2.
        // Detect the presence of stressors.
        int s_DNA_unreplicated  = FLAMEGPU->getVariable<int>("DNA_unreplicated");
        if (s_DNA_unreplicated == 1) {
            if (s_p53 == 1 || s_p73 == 1) {
                s_DNA_unreplicated = 0;
                FLAMEGPU->setVariable<int>("DNA_unreplicated", 0);
            }
        }
        // Update attribute layer 3, part 2.
        // Let the intracellular signalling molecules respond to changes.
        const float s_CDS1_fn = FLAMEGPU->getVariable<float>("CDS1_fn");
        const float s_CDC25C_fn = FLAMEGPU->getVariable<float>("CDC25C_fn");
        const int s_CDS1 = (FLAMEGPU->random.uniform<float>() < s_CDS1_fn && s_DNA_unreplicated == 1) ? 1 : 0;
        const int s_CDC25C = FLAMEGPU->random.uniform<float>() < s_CDC25C_fn && !(s_CDS1 == 1 || s_CHK1 == 1) ? 1 : 0;
        FLAMEGPU->setVariable<int>("CDS1", s_CDS1);
        FLAMEGPU->setVariable<int>("CDC25C", s_CDC25C);

        // The influence of Schwann cells on neuroblast differentiation.
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

        const float nbdiff_jux = FLAMEGPU->environment.getProperty<float>("nbdiff_jux");
        const float nbdiff_amount = FLAMEGPU->environment.getProperty<float>("nbdiff_amount");
        const float nbdiff_para = FLAMEGPU->environment.getProperty<float>("nbdiff_para");
        unsigned int Nscl_count = FLAMEGPU->environment.getProperty<unsigned int>("Nscl_count");
        unsigned int Nnbl_count = FLAMEGPU->environment.getProperty<unsigned int>("Nnbl_count");
        float s_degdiff = FLAMEGPU->getVariable<float>("degdiff");
        if (FLAMEGPU->random.uniform<float>() < step_size * nbdiff_jux * dummy_Nscl / static_cast<float>(dummy_Nnbl + dummy_Nscl)) {
            s_degdiff += nbdiff_amount * step_size;
            if (s_degdiff > 1)
                s_degdiff = 1;
        } else if (FLAMEGPU->random.uniform<float>() < step_size * nbdiff_para * Nscl_count / static_cast<float>(Nnbl_count + Nscl_count)) {
            s_degdiff += nbdiff_amount * step_size;
            if (s_degdiff > 1)
                s_degdiff = 1;
        }
        FLAMEGPU->setVariable<float>("degdiff", s_degdiff);
        FLAMEGPU->setVariable<float>("cycdiff", 1.0f - s_degdiff);

        // Update apoptotic signals.
        // Source 1: CAS.
        // Source 2: Schwann cells.
        // Source 3: Chemotherapy.
        const float nbapop_jux = FLAMEGPU->environment.getProperty<float>("nbapop_jux");
        const float nbapop_para = FLAMEGPU->environment.getProperty<float>("nbapop_para");
        const float P_apopChemo = FLAMEGPU->environment.getProperty<float>("P_apopChemo");
        const float P_apoprp = FLAMEGPU->environment.getProperty<float>("P_apoprp");
        int s_apop_signal = FLAMEGPU->getVariable<int>("apop_signal");
        const unsigned int s_cycle = FLAMEGPU->getVariable<unsigned int>("cycle");
        stress = 0;
        if (s_CAS == 1) {
            s_apop_signal += 1 * step_size;
            stress = 1;
        }
        if (FLAMEGPU->random.uniform<float>() < step_size * nbapop_jux * dummy_Nscl / static_cast<float>(dummy_Nnbl + dummy_Nscl)) {
            s_apop_signal += 1 * step_size;
            stress = 1;
        } else if (FLAMEGPU->random.uniform<float>() < step_size * nbapop_para * Nscl_count / static_cast<float>(Nnbl_count + Nscl_count)) {
            s_apop_signal += 1 * step_size;
            stress = 1;
        }
        const glm::uvec4 cycle_stages = FLAMEGPU->environment.getProperty<glm::uvec4>("cycle_stages");
        if (getChemoState(FLAMEGPU) && cycle_stages[1] < s_cycle && s_cycle < cycle_stages[2] && FLAMEGPU->random.uniform<float>() < P_apopChemo * step_size) {
            s_apop_signal += 1 * step_size;
            stress = 1;
        }
        if (s_apop_signal > 0 && (FLAMEGPU->random.uniform<float>() < P_apoprp * step_size) && stress == 0) {
            s_apop_signal -= 1 * step_size;
        }
        FLAMEGPU->setVariable<int>("apop_signal", s_apop_signal);

        // Update apoptotic status and necrotic status.
        const int apop_critical = FLAMEGPU->environment.getProperty<int>("apop_critical");
        const int s_necro_critical = FLAMEGPU->getVariable<int>("necro_critical");
        if (s_apop_signal >= apop_critical) {
            FLAMEGPU->setVariable<int>("telo", 0);
            FLAMEGPU->setVariable<int>("ALT", 0);
            FLAMEGPU->setVariable<int>("MYCN", 0);
            FLAMEGPU->setVariable<int>("MAPK_RAS", 0);
            FLAMEGPU->setVariable<int>("JAB1", 0);
            FLAMEGPU->setVariable<int>("CHK1", 0);
            FLAMEGPU->setVariable<int>("CDS1", 0);
            FLAMEGPU->setVariable<int>("CDC25C", 0);
            FLAMEGPU->setVariable<int>("ID2", 0);
            FLAMEGPU->setVariable<int>("IAP2", 0);
            FLAMEGPU->setVariable<int>("HIF", 0);
            FLAMEGPU->setVariable<int>("BNIP3", 0);
            FLAMEGPU->setVariable<int>("VEGF", 0);
            FLAMEGPU->setVariable<int>("p53", 0);
            FLAMEGPU->setVariable<int>("p73", 0);
            FLAMEGPU->setVariable<int>("p21", 0);
            FLAMEGPU->setVariable<int>("p27", 0);
            FLAMEGPU->setVariable<int>("Bcl2_Bclxl", 0);
            FLAMEGPU->setVariable<int>("BAK_BAX", 0);
            FLAMEGPU->setVariable<int>("CAS", 0);
            FLAMEGPU->setVariable<int>("mobile", 0);
            FLAMEGPU->setVariable<int>("ATP", 0);
            FLAMEGPU->setVariable<int>("apop", 1);
        } else if (s_necro_signal >= s_necro_critical) {
            FLAMEGPU->setVariable<int>("telo", 0);
            FLAMEGPU->setVariable<int>("ALT", 0);
            FLAMEGPU->setVariable<int>("MYCN", 0);
            FLAMEGPU->setVariable<int>("MAPK_RAS", 0);
            FLAMEGPU->setVariable<int>("JAB1", 0);
            FLAMEGPU->setVariable<int>("CHK1", 0);
            FLAMEGPU->setVariable<int>("CDS1", 0);
            FLAMEGPU->setVariable<int>("CDC25C", 0);
            FLAMEGPU->setVariable<int>("ID2", 0);
            FLAMEGPU->setVariable<int>("IAP2", 0);
            FLAMEGPU->setVariable<int>("HIF", 0);
            FLAMEGPU->setVariable<int>("BNIP3", 0);
            FLAMEGPU->setVariable<int>("VEGF", 0);
            FLAMEGPU->setVariable<int>("p53", 0);
            FLAMEGPU->setVariable<int>("p73", 0);
            FLAMEGPU->setVariable<int>("p21", 0);
            FLAMEGPU->setVariable<int>("p27", 0);
            FLAMEGPU->setVariable<int>("Bcl2_Bclxl", 0);
            FLAMEGPU->setVariable<int>("BAK_BAX", 0);
            FLAMEGPU->setVariable<int>("CAS", 0);
            FLAMEGPU->setVariable<int>("mobile", 0);
            FLAMEGPU->setVariable<int>("ATP", 0);
            FLAMEGPU->setVariable<int>("apop", 0);
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
__device__ __forceinline__ void Neuroblastoma_cell_cycle(flamegpu::DeviceAPI<flamegpu::MessageNone, flamegpu::MessageNone>* FLAMEGPU) {
    // Progress through the neuroblastoma cell cycle.
    // Cell cycle : 0 = G0, 1 = G1 / S, 2 = S / G2, 3 = G2 / M, 4 = division.
    // Regulatory mechanisms:
    //      1. Contact inhibition.
    //      2. ATP availability.
    //      3. Apoptotic / Necrotic status.
    //      4. Extent of differentiation.
    //      5. Gene products.
    const glm::uvec4 cycle_stages = FLAMEGPU->environment.getProperty<glm::uvec4>("cycle_stages");
    const int N_neighbours = FLAMEGPU->environment.getProperty<int>("N_neighbours");
    const unsigned int step_size = FLAMEGPU->environment.getProperty<unsigned int>("step_size");

    unsigned int s_cycle = FLAMEGPU->getVariable<unsigned int>("cycle");
    const int s_neighbours = FLAMEGPU->getVariable<int>("neighbours");
    const int s_ATP = FLAMEGPU->getVariable<int>("ATP");
    const int s_apop = FLAMEGPU->getVariable<int>("apop");
    const int s_necro = FLAMEGPU->getVariable<int>("necro");
    const int s_ID2 = FLAMEGPU->getVariable<int>("ID2");
    const float s_cycdiff = FLAMEGPU->getVariable<float>("cycdiff");
    const int s_MAPK_RAS = FLAMEGPU->getVariable<int>("MAPK_RAS");
    const int s_MYCN = FLAMEGPU->getVariable<int>("MYCN");
    const int s_p21 = FLAMEGPU->getVariable<int>("p21");
    const int s_p27 = FLAMEGPU->getVariable<int>("p27");
    const int s_DNA_unreplicated = FLAMEGPU->getVariable<int>("DNA_unreplicated");
    const int s_CDC25C = FLAMEGPU->getVariable<int>("CDC25C");
    const int s_hypoxia = FLAMEGPU->getVariable<int>("hypoxia");

    // In the cell cycle, 0[12] = G0, 1[6] = G1/S, 2[4] = S/G2, 3[2] = G2/M, 4[0] = division.
    // In the cell cycle, 0-11 = G0, 12-17 = G1/S, 18-21 = S/G2, 22-23 = G2/M, 24+ = division.
    if (s_neighbours <= N_neighbours && s_ATP == 1 && s_apop == 0 && s_necro == 0) {
        if (s_cycle < cycle_stages[0]) {
            if (s_cycle == 0) {
                if (FLAMEGPU->random.uniform<float>() < s_cycdiff) {
                    if (((s_MAPK_RAS == 1 || s_MYCN == 1) && s_p21 == 0 && s_p27 == 0) || s_ID2 == 1) {
                        s_cycle += step_size;
                    }
                }
            } else if (((s_MAPK_RAS == 1 || s_MYCN == 1) && s_p21 == 0 && s_p27 == 0) || s_ID2 == 1) {
                s_cycle += step_size;
                if (s_cycle >= cycle_stages[0] && ((s_MAPK_RAS == 1 && s_p21 == 0 && s_p27 == 0) || s_ID2 == 1) == 0) {
                    s_cycle -= step_size;
                }
            }
        } else if (s_cycle < cycle_stages[1]) {
            if (s_p21 == 0 && s_p27 == 0) {
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
            if (s_cycle >= cycle_stages[2] && s_CDC25C == 0) {
                s_cycle -= step_size;
            }
        } else if (s_cycle < cycle_stages[3]) {
            s_cycle += step_size;
        }
    }
    FLAMEGPU->setVariable<unsigned int>("cycle", s_cycle);
}
__device__ __forceinline__ bool Neuroblastoma_divide(flamegpu::DeviceAPI<flamegpu::MessageNone, flamegpu::MessageNone>* FLAMEGPU) {
    const int s_apop = FLAMEGPU->getVariable<int>("apop");
    const int s_necro = FLAMEGPU->getVariable<int>("necro");
    if (!(s_apop == 0 && s_necro == 0))
        return false;
    // Telomere repair and division of a neuroblastoma cell.
    //  1. If the number of telomere units is below the maximum value, try repairing it.
    //  2. At the end of the cell cycle, the cell divides.
    //      (a)Move it back to the beginning of the cycle.
    //      (b)If it has at least one telomere unit, shorten its telomere.
    const unsigned int s_cycle = FLAMEGPU->getVariable<unsigned int>("cycle");
    int s_telo_count = FLAMEGPU->getVariable<int>("telo_count");
    const int s_telo = FLAMEGPU->getVariable<int>("telo");
    const int s_ALT = FLAMEGPU->getVariable<int>("ALT");
    const int telo_maximum = FLAMEGPU->environment.getProperty<unsigned int>("telo_maximum");
    const float P_telorp = FLAMEGPU->environment.getProperty<float>("P_telorp");
    const unsigned int step_size = FLAMEGPU->environment.getProperty<unsigned int>("step_size");
    const glm::uvec4 cycle_stages = FLAMEGPU->environment.getProperty<glm::uvec4>("cycle_stages");
    if (s_telo_count < telo_maximum) {
        if (s_telo == 1 && (FLAMEGPU->random.uniform<float>() < P_telorp * static_cast<float>(step_size))) {
            s_telo_count += 1;
        } else if (s_ALT == 1 && (FLAMEGPU->random.uniform<float>() < P_telorp * static_cast<float>(step_size))) {
            s_telo_count += 1;
        }
    }
    if (s_cycle >= cycle_stages[3]) {
        FLAMEGPU->setVariable<unsigned int>("cycle", 0);
        if (s_telo_count > 0)
            s_telo_count -= 1;
        FLAMEGPU->setVariable<int>("telo_count", s_telo_count);
        return true;
    }
    FLAMEGPU->setVariable<int>("telo_count", s_telo_count);
    return false;
}

FLAMEGPU_AGENT_FUNCTION(nb_cell_lifecycle, flamegpu::MessageNone, flamegpu::MessageNone) {
    Neuroblastoma_sense(FLAMEGPU);
    Neuroblastoma_cell_cycle(FLAMEGPU);
    if (remove(FLAMEGPU)) {
        return flamegpu::DEAD;  // Kill cell
    } else if (Neuroblastoma_divide(FLAMEGPU)) {
        const glm::vec3 newLoc = drift(FLAMEGPU);
        // Spatial coordinates (integration with imaging biomarkers).
        FLAMEGPU->agent_out.setVariable<glm::vec3>("xyz", newLoc);
        // Data Layer 1 (integration with molecular biomarkers).
        FLAMEGPU->agent_out.setVariable<int>("MYCN_amp", FLAMEGPU->getVariable<int>("MYCN_amp"));
        FLAMEGPU->agent_out.setVariable<int>("TERT_rarngm", FLAMEGPU->getVariable<int>("TERT_rarngm"));
        FLAMEGPU->agent_out.setVariable<int>("ATRX_inact", FLAMEGPU->getVariable<int>("ATRX_inact"));
        FLAMEGPU->agent_out.setVariable<int>("ALT", FLAMEGPU->getVariable<int>("ALT"));
        FLAMEGPU->agent_out.setVariable<int>("ALK", FLAMEGPU->getVariable<int>("ALK"));
        // Data Layer 2 (integration with molecular biomarkers).
        FLAMEGPU->agent_out.setVariable<float>("MYCN_fn00", FLAMEGPU->getVariable<float>("MYCN_fn00"));
        FLAMEGPU->agent_out.setVariable<float>("MYCN_fn10", FLAMEGPU->getVariable<float>("MYCN_fn10"));
        FLAMEGPU->agent_out.setVariable<float>("MYCN_fn01", FLAMEGPU->getVariable<float>("MYCN_fn01"));
        FLAMEGPU->agent_out.setVariable<float>("MYCN_fn11", FLAMEGPU->getVariable<float>("MYCN_fn11"));
        FLAMEGPU->agent_out.setVariable<float>("MAPK_RAS_fn00", FLAMEGPU->getVariable<float>("MAPK_RAS_fn00"));
        FLAMEGPU->agent_out.setVariable<float>("MAPK_RAS_fn10", FLAMEGPU->getVariable<float>("MAPK_RAS_fn10"));
        FLAMEGPU->agent_out.setVariable<float>("MAPK_RAS_fn01", FLAMEGPU->getVariable<float>("MAPK_RAS_fn01"));
        FLAMEGPU->agent_out.setVariable<float>("MAPK_RAS_fn11", FLAMEGPU->getVariable<float>("MAPK_RAS_fn11"));
        FLAMEGPU->agent_out.setVariable<float>("MYCN_fn", FLAMEGPU->getVariable<float>("MYCN_fn"));
        FLAMEGPU->agent_out.setVariable<float>("MAPK_RAS_fn", FLAMEGPU->getVariable<float>("MAPK_RAS_fn"));
        FLAMEGPU->agent_out.setVariable<float>("p53_fn", FLAMEGPU->getVariable<float>("p53_fn"));
        FLAMEGPU->agent_out.setVariable<float>("p73_fn", FLAMEGPU->getVariable<float>("p73_fn"));
        FLAMEGPU->agent_out.setVariable<float>("HIF_fn", FLAMEGPU->getVariable<float>("HIF_fn"));
        // Data Layer 3a (integration with molecular biomarkers).
        FLAMEGPU->agent_out.setVariable<float>("CHK1_fn", FLAMEGPU->getVariable<float>("CHK1_fn"));
        FLAMEGPU->agent_out.setVariable<float>("p21_fn", FLAMEGPU->getVariable<float>("p21_fn"));
        FLAMEGPU->agent_out.setVariable<float>("p27_fn", FLAMEGPU->getVariable<float>("p27_fn"));
        FLAMEGPU->agent_out.setVariable<float>("CDC25C_fn", FLAMEGPU->getVariable<float>("CDC25C_fn"));
        FLAMEGPU->agent_out.setVariable<float>("CDS1_fn", FLAMEGPU->getVariable<float>("CDS1_fn"));
        FLAMEGPU->agent_out.setVariable<float>("ID2_fn", FLAMEGPU->getVariable<float>("ID2_fn"));
        FLAMEGPU->agent_out.setVariable<float>("IAP2_fn", FLAMEGPU->getVariable<float>("IAP2_fn"));
        FLAMEGPU->agent_out.setVariable<float>("BNIP3_fn", FLAMEGPU->getVariable<float>("BNIP3_fn"));
        FLAMEGPU->agent_out.setVariable<float>("JAB1_fn", FLAMEGPU->getVariable<float>("JAB1_fn"));
        FLAMEGPU->agent_out.setVariable<float>("Bcl2_Bclxl_fn", FLAMEGPU->getVariable<float>("Bcl2_Bclxl_fn"));
        FLAMEGPU->agent_out.setVariable<float>("BAK_BAX_fn", FLAMEGPU->getVariable<float>("BAK_BAX_fn"));
        FLAMEGPU->agent_out.setVariable<float>("CAS_fn", FLAMEGPU->getVariable<float>("CAS_fn"));
        FLAMEGPU->agent_out.setVariable<float>("VEGF_fn", FLAMEGPU->getVariable<float>("VEGF_fn"));
        // Initial Conditions.
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
         // Attribute Layer 2.
        FLAMEGPU->agent_out.setVariable<int>("telo", FLAMEGPU->getVariable<int>("telo"));
         // Attribute Layer 3.
        FLAMEGPU->agent_out.setVariable<int>("MYCN", FLAMEGPU->getVariable<int>("MYCN"));
        FLAMEGPU->agent_out.setVariable<int>("MAPK_RAS", FLAMEGPU->getVariable<int>("MAPK_RAS"));
        FLAMEGPU->agent_out.setVariable<int>("JAB1", FLAMEGPU->getVariable<int>("JAB1"));
        FLAMEGPU->agent_out.setVariable<int>("CHK1", FLAMEGPU->getVariable<int>("CHK1"));
        FLAMEGPU->agent_out.setVariable<int>("HIF", FLAMEGPU->getVariable<int>("HIF"));
        FLAMEGPU->agent_out.setVariable<int>("p53", FLAMEGPU->getVariable<int>("p53"));
        FLAMEGPU->agent_out.setVariable<int>("p73", FLAMEGPU->getVariable<int>("p73"));
        FLAMEGPU->agent_out.setVariable<int>("BNIP3", FLAMEGPU->getVariable<int>("BNIP3"));
        FLAMEGPU->agent_out.setVariable<int>("IAP2", FLAMEGPU->getVariable<int>("IAP2"));
        FLAMEGPU->agent_out.setVariable<int>("CDS1", FLAMEGPU->getVariable<int>("CDS1"));
        FLAMEGPU->agent_out.setVariable<int>("Bcl2_Bclxl", FLAMEGPU->getVariable<int>("Bcl2_Bclxl"));
        FLAMEGPU->agent_out.setVariable<int>("BAK_BAX", FLAMEGPU->getVariable<int>("BAK_BAX"));
        FLAMEGPU->agent_out.setVariable<int>("ID2", FLAMEGPU->getVariable<int>("ID2"));
        FLAMEGPU->agent_out.setVariable<int>("CDC25C", FLAMEGPU->getVariable<int>("CDC25C"));
        FLAMEGPU->agent_out.setVariable<int>("p21", FLAMEGPU->getVariable<int>("p21"));
        FLAMEGPU->agent_out.setVariable<int>("p27", FLAMEGPU->getVariable<int>("p27"));
        FLAMEGPU->agent_out.setVariable<int>("VEGF", FLAMEGPU->getVariable<int>("VEGF"));
        FLAMEGPU->agent_out.setVariable<int>("CAS", FLAMEGPU->getVariable<int>("CAS"));
        // Internal
        FLAMEGPU->agent_out.setVariable<float>("force_magnitude", 0);  // This could be left to default init?
        FLAMEGPU->agent_out.setVariable<glm::vec3>("old_xyz", newLoc);
        FLAMEGPU->agent_out.setVariable<float>("move_dist", 0);  // This could be left to default init?
    }
    return flamegpu::ALIVE;
}