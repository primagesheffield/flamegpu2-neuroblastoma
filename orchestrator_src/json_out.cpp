#include <../rapidjson-src/include/rapidjson/writer.h>
#include <../rapidjson-src/include/rapidjson/prettywriter.h>
#include <../rapidjson-src/include/rapidjson/stringbuffer.h>

#include <string>
#include <iostream>
#include <fstream>

#include "structures.h"

void writeOrchestratorOutput(const OrchestratorOutput&out, const std::string &outputFile) {
    rapidjson::StringBuffer s;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer = rapidjson::PrettyWriter<rapidjson::StringBuffer>(s);
    writer.SetIndent('\t', 1);
    // Begin json output object
    writer.StartObject();
    {
        // Version block
        Version v;
        writer.Key("config");
        writer.StartArray();
        {
            for (unsigned int i = 0; i < std::size(v.number); ++i)
                writer.Int(v.number[i]);
            writer.Bool(v.warn_flag);
        }
        writer.EndArray();
        // Primage block
        writer.Key("primage");
        writer.StartObject();
        {
            writer.Key("delta_O2");
            writer.Double(out.delta_O2);
            writer.Key("O2");
            writer.Double(out.O2);
            writer.Key("delta_ecm");
            writer.Double(out.delta_ecm);
            writer.Key("ecm");
            writer.Double(out.ecm);
            writer.Key("material_properties");
            writer.Double(out.material_properties);
            writer.Key("diffusion_coefficient");
            writer.Double(out.diffusion_coefficient);
            writer.Key("total_volume_ratio_updated");
            writer.Double(out.total_volume_ratio_updated);
            writer.Key("cellularity");
            writer.StartArray();
            for (unsigned int i = 0; i < std::size(out.cellularity); ++i)
                writer.Double(out.cellularity[i]);
            writer.EndArray();
            writer.Key("tumour_volume");
            writer.Double(out.tumour_volume);
            writer.Key("ratio_VEGF_NB_SC");
            writer.Double(out.ratio_VEGF_NB_SC);
            writer.Key("nb_telomere_length_mean");
            writer.Double(out.nb_telomere_length_mean);
            writer.Key("nb_telomere_length_sd");
            writer.Double(out.nb_telomere_length_sd);
            writer.Key("sc_telomere_length_mean");
            writer.Double(out.sc_telomere_length_mean);
            writer.Key("sc_telomere_length_sd");
            writer.Double(out.sc_telomere_length_sd);
            writer.Key("nb_necro_signal_mean");
            writer.Double(out.nb_necro_signal_mean);
            writer.Key("nb_necro_signal_sd");
            writer.Double(out.nb_necro_signal_sd);
            writer.Key("nb_apop_signal_mean");
            writer.Double(out.nb_apop_signal_mean);
            writer.Key("nb_apop_signal_sd");
            writer.Double(out.nb_apop_signal_sd);
            writer.Key("sc_necro_signal_mean");
            writer.Double(out.sc_necro_signal_mean);
            writer.Key("sc_necro_signal_sd");
            writer.Double(out.sc_necro_signal_sd);
            writer.Key("sc_apop_signal_mean");
            writer.Double(out.sc_apop_signal_mean);
            writer.Key("sc_apop_signal_sd");
            writer.Double(out.sc_apop_signal_sd);
            writer.Key("extent_of_differentiation_mean");
            writer.Double(out.extent_of_differentiation_mean);
            writer.Key("extent_of_differentiation_sd");
            writer.Double(out.extent_of_differentiation_sd);
        }
        writer.EndObject();
    }
    // End Json file
    writer.EndObject();
    // Perform output
    std::ofstream outStream(outputFile);
    outStream << s.GetString();
    outStream.close();
}
