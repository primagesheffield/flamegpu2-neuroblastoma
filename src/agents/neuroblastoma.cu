#include "header.h"

__device__ __forceinline__ void Neuroblastoma_sense(flamegpu::DeviceAPI<flamegpu::MessageNone, flamegpu::MessageNone> *FLAMEGPU) {
    // Let a neuroblastoma cell respond and potentially adapt to the extracellular environment.
    const int s_apop = FLAMEGPU->getVariable<int>("apop");
    const int s_necro = FLAMEGPU->getVariable<int>("necro");
    const unsigned int step_size = FLAMEGPU->environment.getProperty<unsigned int>("step_size");
    if (s_apop == 0 && s_necro == 0) {
        const float O2 = FLAMEGPU->environment.getProperty<float>("O2");
        const float Cs_O2 = FLAMEGPU->environment.getProperty<float>("Cs_O2");
        const float C50_necro = FLAMEGPU->environment.getProperty<float>("C50_necro");
        const float P_apopChemo = FLAMEGPU->environment.getProperty<float>("P_apopChemo");
        int s_DNA_damage = FLAMEGPU->getVariable<int>("DNA_damage");
        // Detect the presence of stressors.
        // The probability is derived from the hypoxic equilibrium, so it is time-independent.
        const int s_hypoxia = FLAMEGPU->random.uniform<float>() < (1 - O2 * Cs_O2 / (O2 * Cs_O2 + C50_necro)) ? 1 : 0;
        const int s_nutrient = FLAMEGPU->random.uniform<float>() < (1 - O2 * Cs_O2 / (O2 * Cs_O2 + C50_necro)) ? 0 : 1;
        const int CHEMO_ACTIVE = FLAMEGPU->environment.getProperty<int>("CHEMO_ACTIVE");  // Drug delivery is assumed to be instantaneous.
        // Repair any damaged DNA and induce new damages to DNA.
        // 1. p53 and p73 repair damaged DNA.
        // 2. Shortened telomeres make the chromosomes unstable.
        // 3. Hypoxia damages DNA.
        // 4. Chemotherapy damages DNA.
        if (s_DNA_damage == 1) {
            if (FLAMEGPU->getVariable<int>("p53") == 1 || FLAMEGPU->getVariable<int>("p73") == 1)
                s_DNA_damage = 0;
        } else {
            const float P_DNA_damageHypo = FLAMEGPU->environment.getProperty<float>("P_DNA_damageHypo");
            const int telo_critical = FLAMEGPU->environment.getProperty<int>("telo_critical");
            if ((FLAMEGPU->random.uniform<float>() < (1.0f - static_cast<float>(FLAMEGPU->getVariable<int>("telo_count")) / telo_critical) * step_size)
            || (FLAMEGPU->random.uniform<float>() < P_DNA_damageHypo * step_size && s_hypoxia == 1)) {
                s_DNA_damage = 1;
            } else {
                const int CHEMO_OFFSET = CHEMO_ACTIVE ? FLAMEGPU->environment.getProperty<int>("CHEMO_OFFSET") : 0;
                const float chemo0 = FLAMEGPU->environment.getProperty<float>("chemo_effects", CHEMO_OFFSET + 0);
                const float chemo1 = FLAMEGPU->environment.getProperty<float>("chemo_effects", CHEMO_OFFSET + 1);
                const float chemo2 = FLAMEGPU->environment.getProperty<float>("chemo_effects", CHEMO_OFFSET + 2);
                const float chemo3 = FLAMEGPU->environment.getProperty<float>("chemo_effects", CHEMO_OFFSET + 3);
                const float chemo4 = FLAMEGPU->environment.getProperty<float>("chemo_effects", CHEMO_OFFSET + 4);
                const float chemo5 = FLAMEGPU->environment.getProperty<float>("chemo_effects", CHEMO_OFFSET + 5);
                const unsigned int s_cycle = FLAMEGPU->getVariable<unsigned int>("cycle");
                const glm::uvec4 cycle_stages = FLAMEGPU->environment.getProperty<glm::uvec4>("cycle_stages");
                if (CHEMO_ACTIVE && FLAMEGPU->random.uniform<float>() < (chemo0 + chemo1 + chemo2 + chemo3 + chemo4 + chemo5) / 6) {
                    if (cycle_stages[1] < s_cycle && s_cycle < cycle_stages[2] && FLAMEGPU->random.uniform<float>() < P_apopChemo * step_size) {
                        s_DNA_damage = 1;
                    }
                }
            }
        }
        FLAMEGPU->setVariable<int>("DNA_damage", s_DNA_damage);
        FLAMEGPU->setVariable<int>("hypoxia", s_hypoxia);
        FLAMEGPU->setVariable<int>("nutrient", s_nutrient);

        // Update necrotic signals.
        // 1. Signals from necrotic cells.
        // 2. Glycolysis produces lactic acid even if it is successful.
        // 3. Lack of nutrients other than oxygen.
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

        // Update most of the intracellular species.
        // Note that the effects of p53and p73 on HIF are considered before p53and p73 are updated.
        // Chemotherapy inhibits CHK1, JAB1, HIF, MYCN, and p53.It is assumed that drug delivery is instantaneous.
        const int CHEMO_OFFSET = CHEMO_ACTIVE ? FLAMEGPU->environment.getProperty<int>("CHEMO_OFFSET") : 0;
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
        int s_MYCN = FLAMEGPU->random.uniform<float>() < s_MYCN_fn ? 1 : 0;
        if (s_MYCN == 1) {
            const float chemo3 = FLAMEGPU->environment.getProperty<float>("chemo_effects", CHEMO_OFFSET + 3);
            if (CHEMO_ACTIVE && FLAMEGPU->random.uniform<float>() < chemo3) {
                s_MYCN = 0;
            }
        }
        const int s_MAPK_RAS = FLAMEGPU->random.uniform<float>() < s_MAPK_RAS_fn ? 1 : 0;
        int s_JAB1 = FLAMEGPU->random.uniform<float>() < s_JAB1_fn ? 1 : 0;
        if (s_JAB1 == 1) {
            const float chemo1 = FLAMEGPU->environment.getProperty<float>("chemo_effects", CHEMO_OFFSET + 1);
            if (CHEMO_ACTIVE && FLAMEGPU->random.uniform<float>() < chemo1) {
                s_JAB1 = 0;
            }
        }
        int s_CHK1 = (FLAMEGPU->random.uniform<float>() < s_CHK1_fn && s_DNA_damage == 1) || (FLAMEGPU->random.uniform<float>() < s_CHK1_fn && s_DNA_damage == 1 && s_MYCN == 1) ? 1 : 0;
        if (s_CHK1 == 1) {
            const float chemo0 = FLAMEGPU->environment.getProperty<float>("chemo_effects", CHEMO_OFFSET + 0);
            if (CHEMO_ACTIVE && FLAMEGPU->random.uniform<float>() < chemo0) {
                s_CHK1 = 0;
            }
        }
        const int s_ID2 = FLAMEGPU->random.uniform<float>() < s_ID2_fn && s_MYCN == 1 ? 1 : 0;
        const int s_IAP2 = FLAMEGPU->random.uniform<float>() < s_IAP2_fn && s_hypoxia == 1 ? 1 : 0;
        int s_p53 = FLAMEGPU->getVariable<int>("p53");
        int s_p73 = FLAMEGPU->getVariable<int>("p73");
        int s_HIF = ((FLAMEGPU->random.uniform<float>() < s_HIF_fn && s_hypoxia == 1) || (FLAMEGPU->random.uniform<float>() < s_HIF_fn && s_hypoxia == 1 && s_JAB1 == 1)) && !(s_p53 == 1 || s_p73 == 1) ? 1 : 0;
        if (s_HIF == 1) {
            const float chemo2 = FLAMEGPU->environment.getProperty<float>("chemo_effects", CHEMO_OFFSET + 2);
            if (CHEMO_ACTIVE && FLAMEGPU->random.uniform<float>() < chemo2) {
                s_HIF = 0;
            }
        }
        const int s_BNIP3 = FLAMEGPU->random.uniform<float>() < s_BNIP3_fn && s_HIF == 1 ? 1 : 0;
        const int s_VEGF = FLAMEGPU->random.uniform<float>() < s_VEGF_fn && s_HIF == 1 ? 1 : 0;
        const int s_MYCN_amp = FLAMEGPU->getVariable<int>("MYCN_amp");
        s_p53 = ((FLAMEGPU->random.uniform<float>() < s_p53_fn && s_DNA_damage == 1) ||
                (FLAMEGPU->random.uniform<float>() < s_p53_fn && s_DNA_damage == 1 && s_MYCN == 1) ||
                (FLAMEGPU->random.uniform<float>() < s_p53_fn && s_HIF == 1) ||
                (FLAMEGPU->random.uniform<float>() < s_p53_fn&& s_HIF == 1 && s_MYCN == 1)) &&
                !(s_MYCN == 1 && s_MYCN_amp == 1)
                ? 1 : 0;
        if (s_p53 == 1) {
            const float chemo5 = FLAMEGPU->environment.getProperty<float>("chemo_effects", CHEMO_OFFSET + 5);
            if (CHEMO_ACTIVE && FLAMEGPU->random.uniform<float>() < chemo5) {
                s_p53 = 0;
            }
        }
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

        // Try to repair any unreplicated DNA and consider the effects of unreplicated DNA on the intracellular species.
        int s_DNA_unreplicated  = FLAMEGPU->getVariable<int>("DNA_unreplicated");
        if (s_DNA_unreplicated == 1) {
            if (s_p53 == 1 || s_p73 == 1) {
                s_DNA_unreplicated = 0;
                FLAMEGPU->setVariable<int>("DNA_unreplicated", 0);
            }
        }

        const float s_CDS1_fn = FLAMEGPU->getVariable<float>("CDS1_fn");
        const float s_CDC25C_fn = FLAMEGPU->getVariable<float>("CDC25C_fn");
        const int s_CDS1 = (FLAMEGPU->random.uniform<float>() < s_CDS1_fn && s_DNA_unreplicated == 1) ? 1 : 0;
        const int s_CDC25C = FLAMEGPU->random.uniform<float>() < s_CDC25C_fn && !(s_CDS1 == 1 || s_CHK1 == 1) ? 1 : 0;
        FLAMEGPU->setVariable<int>("CDS1", s_CDS1);
        FLAMEGPU->setVariable<int>("CDC25C", s_CDC25C);

        // Differentiation due to stimulation from Schwann cells.
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
        // 1. CAS triggers apoptosis.
        // 2. CAS - independent pathways linking damaged DNA to apoptosis.
        // 3. Schwann cells induce apoptosis.
        const float nbapop_jux = FLAMEGPU->environment.getProperty<float>("nbapop_jux");
        const float nbapop_para = FLAMEGPU->environment.getProperty<float>("nbapop_para");
        const float P_apoprp = FLAMEGPU->environment.getProperty<float>("P_apoprp");
        const float P_DNA_damage_pathways = FLAMEGPU->environment.getProperty<float>("P_DNA_damage_pathways");
        int s_apop_signal = FLAMEGPU->getVariable<int>("apop_signal");
        stress = 0;
        if (s_CAS == 1) {
            s_apop_signal += 1 * step_size;
            stress = 1;
        } else if (s_DNA_damage == 1 && s_ATP == 1 && FLAMEGPU->random.uniform<float>() < step_size * P_DNA_damage_pathways) {
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
        if (s_apop_signal > 0 && (FLAMEGPU->random.uniform<float>() < P_apoprp * step_size) && stress == 0) {
            s_apop_signal -= 1 * step_size;
        }
        FLAMEGPU->setVariable<int>("apop_signal", s_apop_signal);

        // Check if a cell is living, apoptotic, or necrotic.
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
            FLAMEGPU->setVariable<float>("degdiff", 0);
            FLAMEGPU->setVariable<int>("apop_signal", 0);
            FLAMEGPU->setVariable<int>("necro_signal", 0);
            FLAMEGPU->setVariable<int>("telo_count", 0);
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
            FLAMEGPU->setVariable<float>("degdiff", 0);
            FLAMEGPU->setVariable<int>("apop_signal", 0);
            FLAMEGPU->setVariable<int>("necro_signal", 0);
            FLAMEGPU->setVariable<int>("telo_count", 0);
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
    // Regardless of the cell cycle stage, the neuroblast is subjected to several regulatory mechanisms.
    // 1. Basal cycling rate.
    // 2. Contact inhibition.
    // 3. ATP availability.
    // 4. Not being apoptotic/necrotic.
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

    const float P_cycle_nb = FLAMEGPU->environment.getProperty<float>("P_cycle_nb");
    const bool dummy_ncycle = FLAMEGPU->random.uniform<float>() < P_cycle_nb ? true : false;

    // In the cell cycle, 0[12] = G0, 1[6] = G1/S, 2[4] = S/G2, 3[2] = G2/M, 4[0] = division.
    // In the cell cycle, 0-11 = G0, 12-17 = G1/S, 18-21 = S/G2, 22-23 = G2/M, 24+ = division.
    if (dummy_ncycle && s_neighbours <= N_neighbours && s_ATP == 1 && s_apop == 0 && s_necro == 0) {
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
    //        (a)Telomerase: Chemotherapy inhibits TEP1, which is important for telomerase activity. It is assumed that drug delivery is instantaneous.
    //      (b)ALT: This mechanism acts independently of telomerase.
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
            int telo_dummy = 1;
            const int CHEMO_ACTIVE = FLAMEGPU->environment.getProperty<int>("CHEMO_ACTIVE");
            const int CHEMO_OFFSET = FLAMEGPU->environment.getProperty<int>("CHEMO_OFFSET");
            const float chemo4 = FLAMEGPU->environment.getProperty<float>("chemo_effects", CHEMO_OFFSET + 4);
            if (CHEMO_ACTIVE && FLAMEGPU->random.uniform<float>() < chemo4) {
                telo_dummy = 0;
            }
            if (telo_dummy == 1) {
                s_telo_count += 1;
            }
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
FLAMEGPU_AGENT_FUNCTION(output_oxygen_cell, flamegpu::MessageNone, flamegpu::MessageNone) {
    const glm::ivec3 gid = toGrid(FLAMEGPU, FLAMEGPU->getVariable<glm::vec3>("xyz"));
    increment_grid_nb(FLAMEGPU, gid);
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(nb_validation, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("apop") == 0 && FLAMEGPU->getVariable<int>("necro") == 0) {
        ++FLAMEGPU->environment.getMacroProperty<unsigned int>("validation_Nnbl");
    }
    return flamegpu::ALIVE;
}


flamegpu::AgentDescription& defineNeuroblastoma(flamegpu::ModelDescription& model) {
    auto& nb = model.newAgent("Neuroblastoma");
    // The agent's mutation profile.
    // Each mutation is represented by a Boolean variable except ALK, which can take three discrete values.
    {
        nb.newVariable<int>("MYCN_amp");
        nb.newVariable<int>("TERT_rarngm");
        nb.newVariable<int>("ATRX_inact");
        nb.newVariable<int>("ALT");
        nb.newVariable<int>("ALK");  // For ALK, 0 means wild type, 1 means ALK amplification or activation, and 2 means other RAS mutations.
    }
    // Functional activities of various intracellular species.
    // All variables are continuous(0 to 1).
    // Note that MYCN_fnand MAPK_RAS_fn depend on the mutation profile.
    {
        nb.newVariable<float>("MYCN_fn00");
        nb.newVariable<float>("MYCN_fn10");
        nb.newVariable<float>("MYCN_fn01");
        nb.newVariable<float>("MYCN_fn11");
        nb.newVariable<float>("MAPK_RAS_fn00");
        nb.newVariable<float>("MAPK_RAS_fn10");
        nb.newVariable<float>("MAPK_RAS_fn01");
        nb.newVariable<float>("MAPK_RAS_fn11");
        nb.newVariable<float>("MYCN_fn");
        nb.newVariable<float>("MAPK_RAS_fn");
        nb.newVariable<float>("p53_fn");
        nb.newVariable<float>("p73_fn");
        nb.newVariable<float>("HIF_fn");
        nb.newVariable<float>("CHK1_fn");
        nb.newVariable<float>("p21_fn");
        nb.newVariable<float>("p27_fn");
        nb.newVariable<float>("CDC25C_fn");
        nb.newVariable<float>("CDS1_fn");
        nb.newVariable<float>("ID2_fn");
        nb.newVariable<float>("IAP2_fn");
        nb.newVariable<float>("BNIP3_fn");
        nb.newVariable<float>("JAB1_fn");
        nb.newVariable<float>("Bcl2_Bclxl_fn");
        nb.newVariable<float>("BAK_BAX_fn");
        nb.newVariable<float>("CAS_fn");
        nb.newVariable<float>("VEGF_fn");
    }
    // The agent's spatial coordinates, which are continuous variables.
    // By default, it's randomly assigned as the imaging biomarkers are not resolved at this scale.
    // If the initial domain is a sphere rather than a cube, this algorithm ensures that the agents are uniformly distributed throughout
    // the tumour.https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/#using-normally-distributed-random-numbers
    {
        nb.newVariable<float, 3>("xyz");
    }
    // Mechanically related attributes.
    {
        // Fx, Fy, and Fz: Forces in independent directions(kg s - 2 micron).
        nb.newVariable<float, 3>("Fxyz");
        // The agent's overlap with its neighbouring agents.
        nb.newVariable<float>("overlap");
        // The number of agents within the agent's search distance.
        nb.newVariable<int>("neighbours");
        // This Boolean variable indicates whether the agent is mobile.
        nb.newVariable<int>("mobile");
    }
    // Stressors affecting the agent.
    {
        // This Boolean variable indicates whether the agent is hypoxic.
        nb.newVariable<int>("hypoxia");
        // This Boolean variable indicates whether the agent has sufficient nutrients (other than O2).
        nb.newVariable<int>("nutrient");
        // This Boolean variable indicates whether the agent has sufficient energy.
        nb.newVariable<int>("ATP");
        // The total number of telomere units in the agent, a discrete variable ranging from zero to 60.
        nb.newVariable<int>("telo_count");
        // This Boolean variable indicates whether the agent has damaged DNA.
        nb.newVariable<int>("DNA_damage");
        // This Boolean variable indicates whether the agent has unreplicated DNA.
        nb.newVariable<int>("DNA_unreplicated");
    }
    // Attributes about the agent's cycling progress.
    {
        // This flag (continuous, 0 to 4) indicates the agent's position in the cell cycle.
        nb.newVariable<unsigned int>("cycle");
        // Degree of differentiation (continuous between 0 and 1).
        nb.newVariable<float>("degdiff");
        // Probability that the agent enters the cell cycle (G0 to G1).
        nb.newVariable<float>("cycdiff");
    }
    // Attributes about the agent's progress towards apoptosis and necrosis.
    {
        // This Boolean variable indicates if the agent is apoptotic.
        nb.newVariable<int>("apop");
        // The total number of apoptotic signals in the agent.
        nb.newVariable<int>("apop_signal");
        // This Boolean variable indicates if the agent is necrotic.
        nb.newVariable<int>("necro");
        // The total number of necrotic signals in the agent.
        nb.newVariable<int>("necro_signal");
        // The number of necrotic signals required to trigger necrosis in the agent.
        nb.newVariable<int>("necro_critical");
    }
    // Attributes indicating the status of each intracellular species.
    // These Boolean variables are informed by the mutation profile and functional activities of the species.
    {
        nb.newVariable<int>("telo");
        // nb.newVariable<int>("ALT");  // Already declared above
        nb.newVariable<int>("MYCN");
        nb.newVariable<int>("MAPK_RAS");
        nb.newVariable<int>("JAB1");
        nb.newVariable<int>("CHK1");
        nb.newVariable<int>("HIF");
        nb.newVariable<int>("p53");
        nb.newVariable<int>("p73");
        nb.newVariable<int>("BNIP3");
        nb.newVariable<int>("IAP2");
        nb.newVariable<int>("CDS1");
        nb.newVariable<int>("Bcl2_Bclxl");
        nb.newVariable<int>("BAK_BAX");
        nb.newVariable<int>("ID2");
        nb.newVariable<int>("CDC25C");
        nb.newVariable<int>("p21");
        nb.newVariable<int>("p27");
        nb.newVariable<int>("VEGF");
        nb.newVariable<int>("CAS");
    }
    // Internal
    {
        // This is used to provide the dummy_force reduction.
        nb.newVariable<float>("force_magnitude");
        // Old x/y/z and move_dist are used for validating max distance moved.
        nb.newVariable<float, 3>("old_xyz");
        nb.newVariable<float>("move_dist");
    }
    // Agent functions
    {
        nb.newFunction("output_oxygen_cell", output_oxygen_cell);
        auto &t = nb.newFunction("nb_cell_lifecycle", nb_cell_lifecycle);
        t.setAllowAgentDeath(true);
        t.setAgentOutput(nb);
        nb.newFunction("nb_validation", nb_validation);
    }
    return nb;
}

void initNeuroblastoma(flamegpu::HostAPI &FLAMEGPU) {
    auto NB =  FLAMEGPU.agent("Neuroblastoma");
    if (FLAMEGPU.agent("Neuroblastoma").count() != 0)
        return;  // NB agents must have been loaded already

    // Env properties required for initialising NB agents
    const float R_tumour = FLAMEGPU.environment.getProperty<float>("R_tumour");
    const int MYCN_amp = FLAMEGPU.environment.getProperty<int>("MYCN_amp");
    const int TERT_rarngm = FLAMEGPU.environment.getProperty<int>("TERT_rarngm");
    const int ATRX_inact = FLAMEGPU.environment.getProperty<int>("ATRX_inact");
    const int ALT = FLAMEGPU.environment.getProperty<int>("ALT");
    const int ALK = FLAMEGPU.environment.getProperty<int>("ALK");
    const float MYCN_fn00 = FLAMEGPU.environment.getProperty<float>("MYCN_fn00");
    const float MYCN_fn10 = FLAMEGPU.environment.getProperty<float>("MYCN_fn10");
    const float MYCN_fn01 = FLAMEGPU.environment.getProperty<float>("MYCN_fn01");
    const float MYCN_fn11 = FLAMEGPU.environment.getProperty<float>("MYCN_fn11");
    const float MAPK_RAS_fn00 = FLAMEGPU.environment.getProperty<float>("MAPK_RAS_fn00");
    const float MAPK_RAS_fn10 = FLAMEGPU.environment.getProperty<float>("MAPK_RAS_fn10");
    const float MAPK_RAS_fn01 = FLAMEGPU.environment.getProperty<float>("MAPK_RAS_fn01");
    const float MAPK_RAS_fn11 = FLAMEGPU.environment.getProperty<float>("MAPK_RAS_fn11");
    const float p53_fn = FLAMEGPU.environment.getProperty<float>("p53_fn");
    const float p73_fn = FLAMEGPU.environment.getProperty<float>("p73_fn");
    const float HIF_fn = FLAMEGPU.environment.getProperty<float>("HIF_fn");
    const float CHK1_fn = FLAMEGPU.environment.getProperty<float>("CHK1_fn");
    const float p21_fn = FLAMEGPU.environment.getProperty<float>("p21_fn");
    const float p27_fn = FLAMEGPU.environment.getProperty<float>("p27_fn");
    const float CDC25C_fn = FLAMEGPU.environment.getProperty<float>("CDC25C_fn");
    const float CDS1_fn = FLAMEGPU.environment.getProperty<float>("CDS1_fn");
    const float ID2_fn = FLAMEGPU.environment.getProperty<float>("ID2_fn");
    const float IAP2_fn = FLAMEGPU.environment.getProperty<float>("IAP2_fn");
    const float BNIP3_fn = FLAMEGPU.environment.getProperty<float>("BNIP3_fn");
    const float JAB1_fn = FLAMEGPU.environment.getProperty<float>("JAB1_fn");
    const float Bcl2_Bclxl_fn = FLAMEGPU.environment.getProperty<float>("Bcl2_Bclxl_fn");
    const float BAK_BAX_fn = FLAMEGPU.environment.getProperty<float>("BAK_BAX_fn");
    const float CAS_fn = FLAMEGPU.environment.getProperty<float>("CAS_fn");
    const float VEGF_fn = FLAMEGPU.environment.getProperty<float>("VEGF_fn");
    const int cycle = FLAMEGPU.environment.getProperty<int>("cycle");
    const std::array<unsigned int, 4> cycle_stages = FLAMEGPU.environment.getProperty<unsigned int, 4>("cycle_stages");
    const int apop = FLAMEGPU.environment.getProperty<int>("apop");
    const int apop_signal = FLAMEGPU.environment.getProperty<int>("apop_signal");
    const int necro = FLAMEGPU.environment.getProperty<int>("necro");
    const int necro_signal = FLAMEGPU.environment.getProperty<int>("necro_signal");
    const int telo_count = FLAMEGPU.environment.getProperty<int>("telo_count");
    const int histology = FLAMEGPU.environment.getProperty<int>("histology");
    const int gradiff = FLAMEGPU.environment.getProperty<int>("gradiff");

    // Env properties required for calculating agent count
    const float rho_tumour = FLAMEGPU.environment.getProperty<float>("rho_tumour");
    const float V_tumour = FLAMEGPU.environment.getProperty<float>("V_tumour");
    const float theta_sc = FLAMEGPU.environment.getProperty<float>("theta_sc");
    const int total_cell_init = FLAMEGPU.environment.getProperty<int>("total_cell_init");

    const std::array<float, 6> cellularity = FLAMEGPU.environment.getProperty<float, 6>("cellularity");
    const float total_cellularity = (cellularity[0] + cellularity[1] + cellularity[2])/(cellularity[0] + cellularity[1] + cellularity[2] + cellularity[3] + cellularity[4] + cellularity[5]);
    const int orchestrator_time = FLAMEGPU.environment.getProperty<int>("orchestrator_time");

    const float nb_telomere_length_mean = FLAMEGPU.environment.getProperty<float>("nb_telomere_length_mean");
    const float nb_telomere_length_sd = FLAMEGPU.environment.getProperty<float>("nb_telomere_length_sd");
    const float extent_of_differentiation_mean = FLAMEGPU.environment.getProperty<float>("extent_of_differentiation_mean");
    const float extent_of_differentiation_sd = FLAMEGPU.environment.getProperty<float>("extent_of_differentiation_sd");
    const float nb_necro_signal_mean = FLAMEGPU.environment.getProperty<float>("nb_necro_signal_mean");
    const float nb_necro_signal_sd = FLAMEGPU.environment.getProperty<float>("nb_necro_signal_sd");
    const float nb_apop_signal_mean = FLAMEGPU.environment.getProperty<float>("nb_apop_signal_mean");
    const float nb_apop_signal_sd = FLAMEGPU.environment.getProperty<float>("nb_apop_signal_sd");

    const unsigned int NB_COUNT = orchestrator_time == 0 ? (unsigned int)ceil(rho_tumour * V_tumour * total_cellularity) : (unsigned int)ceil(total_cell_init * total_cellularity);
    printf("NB_COUNT: %u\n", NB_COUNT);
    unsigned int validation_Nnbl = 0;
    for (unsigned int i = 0; i < NB_COUNT; ++i) {
        // Decide cell type (living, apop, necro)
        const float cell_rng = FLAMEGPU.random.uniform<float>() * total_cellularity;
        const int IS_APOP = cell_rng >= cellularity[0] && cell_rng < cellularity[0] + cellularity[1];
        const int IS_NECRO = cell_rng >= cellularity[0] + cellularity[1];

        auto agt = NB.newAgent();
        // Spatial coordinates (integration with imaging biomarkers).
        agt.setVariable<glm::vec3>("xyz",
            glm::vec3(-R_tumour + (FLAMEGPU.random.uniform<float>() * 2 * R_tumour),
                -R_tumour + (FLAMEGPU.random.uniform<float>() * 2 * R_tumour),
                -R_tumour + (FLAMEGPU.random.uniform<float>() * 2 * R_tumour)));
        // Data Layer 1 (integration with genetic/molecular biomarkers).
        agt.setVariable<int>("MYCN_amp", MYCN_amp < 0 ? static_cast<int>(FLAMEGPU.random.uniform<float>() < 0.5) : MYCN_amp);
        agt.setVariable<int>("TERT_rarngm", TERT_rarngm < 0 ? static_cast<int>(FLAMEGPU.random.uniform<float>() < 0.5) : TERT_rarngm);
        agt.setVariable<int>("ATRX_inact", ATRX_inact < 0 ? static_cast<int>(FLAMEGPU.random.uniform<float>() < 0.5) : ATRX_inact);
        agt.setVariable<int>("ALT", ALT < 0 ? static_cast<int>(FLAMEGPU.random.uniform<float>() < 0.5) : ALT);
        agt.setVariable<int>("ALK", ALK < 0 ? FLAMEGPU.random.uniform<int>(0, 2) : ALK);  // Random int in range [0, 2]
        // Data Layer 2 (integration with genetic/molecular biomarkers).
        agt.setVariable<float>("MYCN_fn00", MYCN_fn00 < 0 ? FLAMEGPU.random.uniform<float>() : MYCN_fn00);
        agt.setVariable<float>("MYCN_fn10", MYCN_fn10 < 0 ? FLAMEGPU.random.uniform<float>() : MYCN_fn10);
        agt.setVariable<float>("MYCN_fn01", MYCN_fn01 < 0 ? FLAMEGPU.random.uniform<float>() : MYCN_fn01);
        agt.setVariable<float>("MYCN_fn11", MYCN_fn11 < 0 ? FLAMEGPU.random.uniform<float>() : MYCN_fn11);
        agt.setVariable<float>("MAPK_RAS_fn00", MAPK_RAS_fn00 < 0 ? FLAMEGPU.random.uniform<float>() : MAPK_RAS_fn00);
        agt.setVariable<float>("MAPK_RAS_fn10", MAPK_RAS_fn10 < 0 ? FLAMEGPU.random.uniform<float>() : MAPK_RAS_fn10);
        agt.setVariable<float>("MAPK_RAS_fn01", MAPK_RAS_fn01 < 0 ? FLAMEGPU.random.uniform<float>() : MAPK_RAS_fn01);
        agt.setVariable<float>("MAPK_RAS_fn11", MAPK_RAS_fn11 < 0 ? FLAMEGPU.random.uniform<float>() : MAPK_RAS_fn11);
        if (agt.getVariable<int>("MYCN_amp") == 0 && agt.getVariable<int>("ALK") == 0) {
            agt.setVariable<float>("MYCN_fn", agt.getVariable<float>("MYCN_fn00"));
            agt.setVariable<float>("MAPK_RAS_fn", agt.getVariable<float>("MAPK_RAS_fn00"));
        } else if (agt.getVariable<int>("MYCN_amp") == 1 && agt.getVariable<int>("ALK") == 0) {
            agt.setVariable<float>("MYCN_fn", agt.getVariable<float>("MYCN_fn10"));
            agt.setVariable<float>("MAPK_RAS_fn", agt.getVariable<float>("MAPK_RAS_fn10"));
        } else if (agt.getVariable<int>("MYCN_amp") == 0 && agt.getVariable<int>("ALK") == 1) {
            agt.setVariable<float>("MYCN_fn", agt.getVariable<float>("MYCN_fn01"));
            agt.setVariable<float>("MAPK_RAS_fn", agt.getVariable<float>("MAPK_RAS_fn01"));
        } else if (agt.getVariable<int>("MYCN_amp") == 1 && agt.getVariable<int>("ALK") == 1) {
            agt.setVariable<float>("MYCN_fn", agt.getVariable<float>("MYCN_fn11"));
            agt.setVariable<float>("MAPK_RAS_fn", agt.getVariable<float>("MAPK_RAS_fn11"));
        } else if (agt.getVariable<int>("MYCN_amp") == 0 && agt.getVariable<int>("ALK") == 2) {
            agt.setVariable<float>("MYCN_fn", agt.getVariable<float>("MYCN_fn00"));
            agt.setVariable<float>("MAPK_RAS_fn", agt.getVariable<float>("MAPK_RAS_fn01"));
        } else {
            agt.setVariable<float>("MYCN_fn", agt.getVariable<float>("MYCN_fn10"));
            agt.setVariable<float>("MAPK_RAS_fn", agt.getVariable<float>("MAPK_RAS_fn11"));
        }
        agt.setVariable<float>("p53_fn", p53_fn < 0 ? FLAMEGPU.random.uniform<float>() : p53_fn);
        agt.setVariable<float>("p73_fn", p73_fn < 0 ? FLAMEGPU.random.uniform<float>() : p73_fn);
        agt.setVariable<float>("HIF_fn", HIF_fn < 0 ? FLAMEGPU.random.uniform<float>() : HIF_fn);
        // Data Layer 3a (integration with genetic/molecular biomarkers).
        agt.setVariable<float>("CHK1_fn", CHK1_fn < 0 ? FLAMEGPU.random.uniform<float>() : CHK1_fn);
        agt.setVariable<float>("p21_fn", p21_fn < 0 ? FLAMEGPU.random.uniform<float>() : p21_fn);
        agt.setVariable<float>("p27_fn", p27_fn < 0 ? FLAMEGPU.random.uniform<float>() : p27_fn);
        agt.setVariable<float>("CDC25C_fn", CDC25C_fn < 0 ? FLAMEGPU.random.uniform<float>() : CDC25C_fn);
        agt.setVariable<float>("CDS1_fn", CDS1_fn < 0 ? FLAMEGPU.random.uniform<float>() : CDS1_fn);
        agt.setVariable<float>("ID2_fn", ID2_fn < 0 ? FLAMEGPU.random.uniform<float>() : ID2_fn);
        agt.setVariable<float>("IAP2_fn", IAP2_fn < 0 ? FLAMEGPU.random.uniform<float>() : IAP2_fn);
        agt.setVariable<float>("BNIP3_fn", BNIP3_fn < 0 ? FLAMEGPU.random.uniform<float>() : BNIP3_fn);
        agt.setVariable<float>("JAB1_fn", JAB1_fn < 0 ? FLAMEGPU.random.uniform<float>() : JAB1_fn);
        agt.setVariable<float>("Bcl2_Bclxl_fn", Bcl2_Bclxl_fn < 0 ? FLAMEGPU.random.uniform<float>() : Bcl2_Bclxl_fn);
        agt.setVariable<float>("BAK_BAX_fn", BAK_BAX_fn < 0 ? FLAMEGPU.random.uniform<float>() : BAK_BAX_fn);
        agt.setVariable<float>("CAS_fn", CAS_fn < 0 ? FLAMEGPU.random.uniform<float>() : CAS_fn);
        agt.setVariable<float>("VEGF_fn", VEGF_fn < 0 ? FLAMEGPU.random.uniform<float>() : VEGF_fn);
        // Initial conditions.
        agt.setVariable<glm::vec3>("Fxyz", glm::vec3(0));
        agt.setVariable<float>("overlap", 0);
        agt.setVariable<int>("neighbours", 0);
        agt.setVariable<int>("mobile", 1);
        agt.setVariable<int>("ATP", 1);
        if (cycle >= 0) {
            agt.setVariable<unsigned int>("cycle", static_cast<unsigned int>(cycle));
        } else {
            // Weird init, because Py model has uniform chance per stage
            // uniform chance within stage
            const int stage = FLAMEGPU.random.uniform<int>(0, 3);  // Random int in range [0, 3]
            const unsigned int stage_start = stage == 0 ? 0 : cycle_stages[stage - 1];
            const unsigned int stage_extra = static_cast<unsigned int>(FLAMEGPU.random.uniform<float>() * static_cast<float>(cycle_stages[stage] - stage_start));
            agt.setVariable<unsigned int>("cycle", stage_start + stage_extra);
        }
        agt.setVariable<int>("apop", IS_APOP);
        agt.setVariable<int>("necro", IS_NECRO);
        validation_Nnbl += IS_APOP || IS_NECRO ? 0 : 1;
        agt.setVariable<int>("necro_critical", FLAMEGPU.random.uniform<int>(3, 168));  // Random int in range [3, 168]
        if (IS_APOP || IS_NECRO) {
            agt.setVariable<float>("degdiff", 0);
            agt.setVariable<int>("apop_signal", 0);
            agt.setVariable<int>("necro_signal", 0);
            agt.setVariable<int>("telo_count", 0);
        } else if (orchestrator_time == 0) {
            agt.setVariable<int>("necro_signal", necro_signal < 0 ? 0 : necro_signal);
            agt.setVariable<int>("apop_signal", apop_signal < 0 ? 0 : apop_signal);
            if (telo_count < 0) {
                agt.setVariable<int>("telo_count", FLAMEGPU.random.uniform<int>(35, 45));  // Random int in range [35, 45]
            } else if (telo_count == 1) {
                agt.setVariable<int>("telo_count", FLAMEGPU.random.uniform<int>(47, 60));  // Random int in range [47, 60]
            } else if (telo_count == 2) {
                agt.setVariable<int>("telo_count", FLAMEGPU.random.uniform<int>(21, 33));  // Random int in range [21, 33]
            } else if (telo_count == 3) {
                agt.setVariable<int>("telo_count", FLAMEGPU.random.uniform<int>(34, 46));  // Random int in range [34, 46]
            } else {
                agt.setVariable<int>("telo_count", telo_count);
            }

            if (histology == 0) {
                if (gradiff == 0) {
                    agt.setVariable<float>("degdiff", 0);
                } else if (gradiff == 1) {
                    agt.setVariable<float>("degdiff", FLAMEGPU.random.uniform<float>() / 5.0f);
                } else if (gradiff == 2) {
                    agt.setVariable<float>("degdiff", 0.2f + (FLAMEGPU.random.uniform<float>() / 5.0f));
                }
            } else if (histology == 2) {
                const float dummy = FLAMEGPU.random.uniform<float>();
                if (dummy < 0.33) {
                    if (gradiff == 0) {
                        agt.setVariable<float>("degdiff", 0);
                    } else if (gradiff == 1) {
                        agt.setVariable<float>("degdiff", FLAMEGPU.random.uniform<float>() / 5.0f);
                    } else if (gradiff == 2) {
                        agt.setVariable<float>("degdiff", 0.2f + (FLAMEGPU.random.uniform<float>() / 5.0f));
                    }
                } else if (dummy < 0.66f) {
                    agt.setVariable<float>("degdiff", 0.4f + (FLAMEGPU.random.uniform<float>() / 5.0f));
                } else {
                    agt.setVariable<float>("degdiff", 0.6f + (FLAMEGPU.random.uniform<float>() / 5.0f));
                }
            } else if (histology == 3) {
                agt.setVariable<float>("degdiff", 0.4f + (FLAMEGPU.random.uniform<float>() / 5.0f));
            } else if (histology == 5) {
                agt.setVariable<float>("degdiff", 0.6f + (FLAMEGPU.random.uniform<float>() / 5.0f));
            } else if (histology == 6) {
                agt.setVariable<float>("degdiff", 0.8f + (FLAMEGPU.random.uniform<float>() / 5.0f));
            }
        } else {
            agt.setVariable<int>("necro_signal", std::max(0, static_cast<int>((FLAMEGPU.random.normal<float>() * nb_apop_signal_sd) + nb_apop_signal_mean)));
            agt.setVariable<int>("apop_signal", std::max(0, static_cast<int>((FLAMEGPU.random.normal<float>() * nb_necro_signal_sd) + nb_necro_signal_mean)));
            agt.setVariable<int>("telo_count", std::max(0, static_cast<int>((FLAMEGPU.random.normal<float>() * nb_telomere_length_sd) + nb_telomere_length_mean)));
            agt.setVariable<float>("degdiff", std::max(0.0f, (FLAMEGPU.random.normal<float>()* extent_of_differentiation_sd) + extent_of_differentiation_mean));
        }
        agt.setVariable<float>("cycdiff", 1.0f - agt.getVariable<float>("degdiff"));
        // Attribute Layer 1.
        agt.setVariable<int>("hypoxia", 0);
        agt.setVariable<int>("nutrient", 1);
        agt.setVariable<int>("DNA_damage", 0);
        agt.setVariable<int>("DNA_unreplicated", 0);
        // Attribute Layer 2.
        agt.setVariable<int>("telo", (agt.getVariable<int>("MYCN_amp") == 1 || agt.getVariable<int>("TERT_rarngm") == 1) ? 1 : 0);
        agt.setVariable<int>("ALT", agt.getVariable<int>("ALT") == 0 && ATRX_inact == 1 ? 1 : agt.getVariable<int>("ALT"));
        // Attribute Layer 3. (Could tweak these so RNG is always final component of condition to maybe improve perf)
        const float a_MYCN_fn = agt.getVariable<float>("MYCN_fn");
        const float a_MAPK_RAS_fn = agt.getVariable<float>("MAPK_RAS_fn");
        const float a_JAB1_fn = agt.getVariable<float>("JAB1_fn");
        const float a_CHK1_fn = agt.getVariable<float>("CHK1_fn");
        const int a_DNA_damage = agt.getVariable<int>("DNA_damage");
        const float a_CDS1_fn = agt.getVariable<float>("CDS1_fn");
        const int a_DNA_unreplicated = agt.getVariable<int>("DNA_unreplicated");
        const float a_CDC25C_fn = agt.getVariable<float>("CDC25C_fn");
        const float a_ID2_fn = agt.getVariable<float>("ID2_fn");
        const float a_IAP2_fn = agt.getVariable<float>("IAP2_fn");
        const int a_hypoxia = agt.getVariable<int>("hypoxia");
        const float a_HIF_fn = agt.getVariable<float>("HIF_fn");
        const float a_BNIP3_fn = agt.getVariable<float>("BNIP3_fn");
        const float a_VEGF_fn = agt.getVariable<float>("VEGF_fn");
        const float a_p53_fn = agt.getVariable<float>("p53_fn");
        const int a_MYCN_amp = agt.getVariable<int>("MYCN_amp");
        const float a_p73_fn = agt.getVariable<float>("p73_fn");
        const float a_p21_fn = agt.getVariable<float>("p21_fn");
        const float a_p27_fn = agt.getVariable<float>("p27_fn");
        const float a_Bcl2_Bclxl_fn = agt.getVariable<float>("Bcl2_Bclxl_fn");
        const float a_BAK_BAX_fn = agt.getVariable<float>("BAK_BAX_fn");
        const float a_CAS_fn = agt.getVariable<float>("CAS_fn");
        const int a_ATP = agt.getVariable<int>("ATP");
        const int a_Bcl2_Bclxl = agt.getVariable<int>("Bcl2_Bclxl");

        const int a_MYCN = FLAMEGPU.random.uniform<float>() < a_MYCN_fn ? 1 : 0;
        const int a_MAPK_RAS = FLAMEGPU.random.uniform<float>() < a_MAPK_RAS_fn ? 1 : 0;
        const int a_JAB1 = FLAMEGPU.random.uniform<float>() < a_JAB1_fn ? 1 : 0;
        const int a_CHK1 = (FLAMEGPU.random.uniform<float>() < a_CHK1_fn && a_DNA_damage == 1) || (a_MYCN == 1 && FLAMEGPU.random.uniform<float>() < a_CHK1_fn && a_DNA_damage == 1) ? 1 : 0;  // Unusual case here, if MYCN is on, second chance
        const int a_CDS1 = (FLAMEGPU.random.uniform<float>() < a_CDS1_fn && a_DNA_unreplicated == 1) ? 1 : 0;
        const int a_IAP2 = (FLAMEGPU.random.uniform<float>() < a_IAP2_fn && a_hypoxia == 1) ? 1 : 0;
        const int a_HIF = ((FLAMEGPU.random.uniform<float>() < a_HIF_fn && a_hypoxia == 1) || (FLAMEGPU.random.uniform<float>() < a_HIF_fn && a_hypoxia == 1 && a_JAB1 == 1)) && !(agt.getVariable<int>("p53") == 1 || agt.getVariable<int>("p73") == 1) ? 1 : 0;  // Unusual case here, if JAB1 is on, second chance
        const int a_BNIP3 = (FLAMEGPU.random.uniform<float>() < a_BNIP3_fn && a_HIF == 1) ? 1 : 0;
        const int a_p53 = ((FLAMEGPU.random.uniform<float>() < a_p53_fn && a_DNA_damage == 1) ||
                          (FLAMEGPU.random.uniform<float>() < a_p53_fn && a_DNA_damage == 1 && a_MYCN == 1) ||
                          (FLAMEGPU.random.uniform<float>() < a_p53_fn && a_HIF == 1) ||
                          (FLAMEGPU.random.uniform<float>() < a_p53_fn && a_HIF == 1 && a_MYCN == 1)) &&
                          !(a_MYCN == 1 && a_MYCN_amp == 1)
                          ? 1 : 0;
        const int a_p73 = ((FLAMEGPU.random.uniform<float>() < a_p73_fn && a_CHK1 == 1) ||
                          (FLAMEGPU.random.uniform<float>() < a_p73_fn && a_HIF == 1))
                          ? 1 : 0;
        const int a_BAK_BAX = (((FLAMEGPU.random.uniform<float>() < a_BAK_BAX_fn && a_hypoxia == 1) || (FLAMEGPU.random.uniform<float>() < a_BAK_BAX_fn && a_p53 == 1) || (FLAMEGPU.random.uniform<float>() < a_BAK_BAX_fn && a_p73 == 1)) && !(a_Bcl2_Bclxl == 1 || a_IAP2 == 1)) ? 1 : 0;

        agt.setVariable<int>("MYCN", a_MYCN);
        agt.setVariable<int>("MAPK_RAS", a_MAPK_RAS);
        agt.setVariable<int>("JAB1", a_JAB1);
        agt.setVariable<int>("CHK1", a_CHK1);
        agt.setVariable<int>("CDS1", a_CDS1);
        agt.setVariable<int>("CDC25C", (FLAMEGPU.random.uniform<float>() < a_CDC25C_fn && !(a_CDS1 == 1 || a_CHK1 == 1)) ? 1 : 0);
        agt.setVariable<int>("ID2", (FLAMEGPU.random.uniform<float>() < a_ID2_fn && a_MYCN == 1) ? 1 : 0);
        agt.setVariable<int>("IAP2", a_IAP2);
        agt.setVariable<int>("HIF", a_HIF);
        agt.setVariable<int>("BNIP3", a_BNIP3);
        agt.setVariable<int>("VEGF", (FLAMEGPU.random.uniform<float>() < a_VEGF_fn && a_HIF == 1) ? 1 : 0);
        agt.setVariable<int>("p53", a_p53);
        agt.setVariable<int>("p73", a_p73);
        agt.setVariable<int>("p21", (((FLAMEGPU.random.uniform<float>() < a_p21_fn && a_HIF == 1) || (FLAMEGPU.random.uniform<float>() < a_p21_fn && a_p53 == 1)) && !(a_MAPK_RAS == 1 || a_MYCN == 1)) ? 1 : 0);
        agt.setVariable<int>("p27", (((FLAMEGPU.random.uniform<float>() < a_p27_fn && a_HIF == 1) || (FLAMEGPU.random.uniform<float>() < a_p27_fn && a_p53 == 1)) && !(a_MAPK_RAS == 1 || a_MYCN == 1)) ? 1 : 0);
        agt.setVariable<int>("Bcl2_Bclxl", (FLAMEGPU.random.uniform<float>() < a_Bcl2_Bclxl_fn && !(a_BNIP3 == 1 || a_p53 == 1 || a_p73 == 1)) ? 1 : 0);
        agt.setVariable<int>("BAK_BAX", a_BAK_BAX);
        agt.setVariable<int>("CAS", ((FLAMEGPU.random.uniform<float>() < a_CAS_fn && a_BAK_BAX == 1) || (FLAMEGPU.random.uniform<float>() < a_CAS_fn && a_hypoxia == 1)) && a_ATP == 1 ? 1 : 0);

        // Internal.
        agt.setVariable<float>("force_magnitude", 0);
    }
    FLAMEGPU.environment.setProperty<unsigned int>("validation_Nnbl", validation_Nnbl);
}
