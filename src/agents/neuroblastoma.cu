#include "header.h"

FLAMEGPU_AGENT_FUNCTION(nb_cell_lifecycle, flamegpu::MessageNone, flamegpu::MessageNone) {
   // @todo 
}
FLAMEGPU_AGENT_FUNCTION(apply_nb_force, flamegpu::MessageNone, flamegpu::MessageNone) {
    // @todo 
}
FLAMEGPU_AGENT_FUNCTION(output_nb_location, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    // @todo 
}
FLAMEGPU_AGENT_FUNCTION(calculate_nb_force, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    // @todo 
}
FLAMEGPU_AGENT_FUNCTION(output_oxygen_cell, flamegpu::MessageNone, flamegpu::MessageNone) {
    // @todo 
}


flamegpu::AgentDescription& defineNeuroblastoma(flamegpu::ModelDescription& model) {
    auto& nb = model.newAgent("Neuroblastoma");
    // Spatial coordinates (integration with imaging biomarkers).
    {
        nb.newVariable<glm::vec3>("xyz");
    }
    // Data Layer 1 (integration with molecular biomarkers).
    {
        nb.newVariable<int>("MYCN_amp");
        nb.newVariable<int>("TERT_rarngm");
        nb.newVariable<int>("ATRX_inact");
        nb.newVariable<int>("ALT");
        nb.newVariable<int>("ALK");
    }
    // Data Layer 2 (integration with molecular biomarkers).
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
    }
    // Data Layer 3 (integration with molecular biomarkers).
    {
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
    // Initial Conditions
    {
        // Fx, Fy, and Fz are forces in independent directions (kg s-2 micron).
        nb.newVariable<glm::vec3>("Fxyz");
        // overlap is the cell's overlap with its neighbouring cells.
        nb.newVariable<float>("overlap");
        // neighbours is the number of cells within the cell's search distance.
        nb.newVariable<int>("neighbours");
        // mobile indicates whether the neuroblast is mobile.
        nb.newVariable<int>("mobile");
        // ATP indicates whether the neuroblast has sufficient energy.
        nb.newVariable<int>("ATP");
        // cycle is a flag (continuous, 0 to 4) indicating the cell's position in the cell cycle.
        nb.newVariable<unsigned int>("cycle");
        // apop indicates if the cell is apoptotic. It is a Boolean variable.
        nb.newVariable<int>("apop");
        // apop_signal is the total number of apoptotic signals. Maximum is 3 (Elmore, 2007).
        nb.newVariable<int>("apop_signal");
        // necro indicates if the cell is necrotic. It is a Boolean variable.
        nb.newVariable<int>("necro");
        // necro_signal is the total number of necrotic signals. Maximum is 168 (Warren et al., 2016).
        nb.newVariable<int>("necro_signal");
        // necro_critical is the number of necrotic signals required to trigger necrosis in a neuroblast. It is between 3 and 168, inclusive (Warren et al., 2016).
        nb.newVariable<int>("necro_critical");
        // telo_count is the total number of telomere units. The maximum is 60 (Hayflick et al., 1961).
        nb.newVariable<int>("telo_count");
        // the degree of differentiation (0 means an undifferentiated neuroblast, 1 means a poorly differentiated one, and 2 means a fully differentiated one).
        nb.newVariable<float>("degdiff");
        // the probability of the neuroblast entering the cell cycle (G0 to G1), determined by degdiff and time-independent, i.e. at equilibrium.
        nb.newVariable<float>("cycdiff");
    }
    // Attribute Layer 1.
    // MYCN's effects on p53 depend on the former's amplification status(Tang et al., 2006).
    // Note that the effects of p53 and p73 on HIF are considered before p53 and p73 are updated.
    {
        nb.newVariable<int>("hypoxia");
        nb.newVariable<int>("nutrient");
        nb.newVariable<int>("DNA_damage");
        nb.newVariable<int>("DNA_unreplicated");
    }
    // Attribute Layer 2.
    {
        nb.newVariable<int>("telo");
    }
    // Attribute Layer 3.
    {
        nb.newVariable<int>("MYCN");
        nb.newVariable<int>("MAPK_RAS");
        nb.newVariable<int>("JAB1");
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
        nb.newVariable<glm::vec3>("old_xyz");
        nb.newVariable<float>("move_dist");
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
    const float cycle = FLAMEGPU.environment.getProperty<float>("cycle");
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
    const float cellularity = FLAMEGPU.environment.getProperty<float>("cellularity");
    const float theta_sc = FLAMEGPU.environment.getProperty<float>("theta_sc");

    const unsigned int NB_COUNT = (unsigned int)ceil(rho_tumour * V_tumour * cellularity * (1 - theta_sc));

    for (unsigned int i = 0; i < NB_COUNT; ++i) {
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
        agt.setVariable<int>("ALK", ALK < 0 ? FLAMEGPU.random.uniform<int>(0, 2) : ALK); // Random int in range [0, 2]
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
        agt.setVariable<float>("Bcl2_Bclxl_fn", Bcl2_Bclxl_fn < 0 ? FLAMEGPU.random.uniform<float>() : Bcl2_Bclxl_fn);
        agt.setVariable<float>("VEGF_fn", VEGF_fn < 0 ? FLAMEGPU.random.uniform<float>() : VEGF_fn);
        //Initial conditions.
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
            const int stage = FLAMEGPU.random.uniform<int>(0, 3); // Random int in range [0, 3]
            const unsigned int stage_start = stage == 0 ? 0 : cycle_stages[stage - 1];
            const unsigned int stage_extra = static_cast<unsigned int>(FLAMEGPU.random.uniform<float>() * static_cast<float>(cycle_stages[stage] - stage_start));
            agt.setVariable<unsigned int>("cycle", stage_start + stage_extra);
        }
        agt.setVariable<int>("apop", apop < 0 ? 0 : apop);
        agt.setVariable<int>("apop_signal", apop_signal < 0 ? 0 : apop_signal);
        agt.setVariable<int>("necro", necro < 0 ? 0 : necro);
        agt.setVariable<int>("necro_signal", necro_signal < 0 ? 0 : necro_signal);
        agt.setVariable<int>("necro_critical", FLAMEGPU.random.uniform<int>(3, 168)); // Random int in range [3, 168]
        if (telo_count < 0) {
            agt.setVariable<int>("telo_count", FLAMEGPU.random.uniform<int>(25, 35)); // Random int in range [25, 35]
        } else if (telo_count == 1) {
            agt.setVariable<int>("telo_count", FLAMEGPU.random.uniform<int>(41, 60)); // Random int in range [41, 60]
        } else if (telo_count == 2) {
            agt.setVariable<int>("telo_count", FLAMEGPU.random.uniform<int>(1, 20)); // Random int in range [1, 20]
        } else if (telo_count == 3) {
            agt.setVariable<int>("telo_count", FLAMEGPU.random.uniform<int>(21, 40)); // Random int in range [21, 40]
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
        agt.setVariable<float>("cycdiff", 1.0f - agt.getVariable<float>("degdiff"));
        // Attribute Layer 1.
        agt.setVariable<int>("hypoxia", 0);
        agt.setVariable<int>("nutrient", 1);
        agt.setVariable<int>("DNA_damage", 0);
        agt.setVariable<int>("DNA_unreplicated", 0);
        // Attribute Layer 2.
        agt.setVariable<int>("telo", (agt.getVariable<float>("MYCN_amp") == 1 || agt.getVariable<float>("TERT_rarngm") == 1) ? 1 : 0);
        agt.setVariable<int>("ALT", agt.getVariable<float>("ALT") == 0 && ATRX_inact == 1 ? 1 : agt.getVariable<float>("ALT"));
        // Attribute Layer 3. (Could tweak these so RNG is always final component of condition to maybe improve perf)
        const float a_MYCN_fn = agt.getVariable<float>("MYCN_fn");
        const float a_MAPK_RAS_fn = agt.getVariable<float>("MAPK_RAS_fn");
        const float a_JAB1_fn = agt.getVariable<float>("JAB1_fn");
        const float a_CHK1_fn = agt.getVariable<float>("CHK1_fn");
        const int a_DNA_damage = agt.getVariable<int>("DNA_damage");
        const int a_MYCN = agt.getVariable<int>("MYCN");
        const int a_CDS1_fn = agt.getVariable<int>("CDS1_fn");
        const int a_DNA_unreplicated = agt.getVariable<int>("DNA_unreplicated");
        const int a_CDC25C_fn = agt.getVariable<int>("CDC25C_fn");
        const int a_CDS1 = agt.getVariable<int>("CDS1");
        const int a_CHK1 = agt.getVariable<int>("CHK1");
        const int a_ID2_fn = agt.getVariable<int>("ID2_fn");
        const int a_IAP2_fn = agt.getVariable<int>("IAP2_fn");
        const int a_hypoxia = agt.getVariable<int>("hypoxia");
        const int a_HIF_fn = agt.getVariable<int>("HIF_fn");
        const int a_BNIP3_fn = agt.getVariable<int>("BNIP3_fn");
        const int a_HIF = agt.getVariable<int>("HIF");
        const int a_VEGF_fn = agt.getVariable<int>("VEGF_fn");
        const int a_p53_fn = agt.getVariable<int>("p53_fn");
        const int a_JAB1 = agt.getVariable<int>("JAB1");
        const int a_p53 = agt.getVariable<int>("p53");
        const int a_p73 = agt.getVariable<int>("p73");
        const int a_MYCN_amp = agt.getVariable<int>("MYCN_amp");
        const int a_p73_fn = agt.getVariable<int>("p73_fn");
        const int a_p21_fn = agt.getVariable<int>("p21_fn");
        const int a_MAPK_RAS = agt.getVariable<int>("MAPK_RAS");
        const int a_p27_fn = agt.getVariable<int>("p27_fn");
        const int a_Bcl2_Bclxl_fn = agt.getVariable<int>("Bcl2_Bclxl_fn");
        const int a_BNIP3 = agt.getVariable<int>("BNIP3");
        const int a_BAK_BAX_fn = agt.getVariable<int>("BAK_BAX_fn");
        const int a_IAP2 = agt.getVariable<int>("IAP2");
        const int a_CAS_fn = agt.getVariable<int>("CAS_fn");
        const int a_BAK_BAX = agt.getVariable<int>("BAK_BAX");
        const int a_ATP = agt.getVariable<int>("ATP");

        agt.setVariable<int>("MYCN", FLAMEGPU.random.uniform<float>() < a_MYCN_fn ? 1 : 0);
        agt.setVariable<int>("MAPK_RAS", FLAMEGPU.random.uniform<float>() < a_MAPK_RAS_fn ? 1 : 0);
        agt.setVariable<int>("JAB1", FLAMEGPU.random.uniform<float>() < a_JAB1_fn ? 1 : 0);
        agt.setVariable<int>("CHK1", (FLAMEGPU.random.uniform<float>() < a_CHK1_fn && a_DNA_damage == 1) || (a_MYCN == 1 && FLAMEGPU.random.uniform<float>() < a_CHK1_fn && a_DNA_damage == 1) ? 1 : 0);  // Unusual case here, if MYCN is on, second chance
        agt.setVariable<int>("CDS1", (FLAMEGPU.random.uniform<float>() < a_CDS1_fn && a_DNA_unreplicated == 1) ? 1 : 0);
        agt.setVariable<int>("CDC25C", (FLAMEGPU.random.uniform<float>() < a_CDC25C_fn && !(a_CDS1 == 1 || a_CHK1 == 1)) ? 1 : 0);
        agt.setVariable<int>("ID2", (FLAMEGPU.random.uniform<float>() < a_ID2_fn && a_MYCN == 1) ? 1 : 0);
        agt.setVariable<int>("IAP2", (FLAMEGPU.random.uniform<float>() < a_IAP2_fn && a_hypoxia == 1) ? 1 : 0);
        agt.setVariable<int>("HIF", ((FLAMEGPU.random.uniform<float>() < a_HIF_fn && a_hypoxia == 1) || (FLAMEGPU.random.uniform<float>() < a_HIF_fn && a_hypoxia == 1 && a_JAB1 == 1)) && !(a_p53 == 1 || a_p73 == 1) ? 1 : 0);  // Unusual case here, if JAB1 is on, second chance
        agt.setVariable<int>("BNIP3", (FLAMEGPU.random.uniform<float>() < a_BNIP3_fn && a_HIF == 1) ? 1 : 0);
        agt.setVariable<int>("VEGF", (FLAMEGPU.random.uniform<float>() < a_VEGF_fn && a_HIF == 1) ? 1 : 0);
        agt.setVariable<int>("p53", ((FLAMEGPU.random.uniform<float>() < a_p53_fn && a_DNA_damage == 1) || (FLAMEGPU.random.uniform<float>() < a_p53_fn && a_DNA_damage == 1 && a_MYCN == 1) || (FLAMEGPU.random.uniform<float>() < a_p53_fn && a_HIF == 1) || (FLAMEGPU.random.uniform<float>() < a_p53_fn && a_HIF == 1 && a_MYCN == 1)) && !(a_MYCN == 1 && a_MYCN_amp == 1) ? 1 : 0);
        agt.setVariable<int>("p73", ((FLAMEGPU.random.uniform<float>() < a_p73_fn && a_CHK1 == 1) || (FLAMEGPU.random.uniform<float>() < a_p73_fn && a_HIF == 1)) ? 1 : 0);
        agt.setVariable<int>("p21", (((FLAMEGPU.random.uniform<float>() < a_p21_fn && a_HIF == 1) || (FLAMEGPU.random.uniform<float>() < a_p21_fn && a_p53 == 1)) && !(a_MAPK_RAS == 1 || a_MYCN == 1)) ? 1 : 0);
        agt.setVariable<int>("p27", (((FLAMEGPU.random.uniform<float>() < a_p27_fn && a_HIF == 1) || (FLAMEGPU.random.uniform<float>() < a_p27_fn && a_p53 == 1)) && !(a_MAPK_RAS == 1 || a_MYCN == 1)) ? 1 : 0);
        agt.setVariable<int>("Bcl2_Bclxl", (FLAMEGPU.random.uniform<float>() < a_Bcl2_Bclxl_fn && !(a_BNIP3 == 1 || a_p53 == 1 || a_p73 == 1)) ? 1 : 0);
        agt.setVariable<int>("BAK_BAX", (((FLAMEGPU.random.uniform<float>() < a_BAK_BAX_fn && a_hypoxia == 1) || (FLAMEGPU.random.uniform<float>() < a_BAK_BAX_fn && a_p53 == 1) || (FLAMEGPU.random.uniform<float>() < a_BAK_BAX_fn && a_p73 == 1)) && !(a_Bcl2_Bclxl == 1 || a_IAP2 == 1)) ? 1 : 0);
        agt.setVariable<int>("CAS", ((FLAMEGPU.random.uniform<float>() < a_CAS_fn && a_BAK_BAX == 1) || (FLAMEGPU.random.uniform<float>() < a_CAS_fn && a_hypoxia == 1)) && a_ATP == 1 ? 1 : 0);

        // Internal.
        agt.setVariable<float>("force_magnitude", 0);
    }
}