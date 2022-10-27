#include <../rapidjson-src/include/rapidjson/writer.h>
#include <../rapidjson-src/include/rapidjson/prettywriter.h>
#include <../rapidjson-src/include/rapidjson/stringbuffer.h>

#include <string>
#include <iostream>
#include <fstream>
#include <cassert>

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
        writer.Key("version");
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
            // Primage expects array in different order to what we implemented
            // So remap it here
            assert(std::size(out.cellularity) == 6);
            writer.Double(out.cellularity[0]);  // NB Living
            writer.Double(out.cellularity[3]);  // SC Living
            writer.Double(out.cellularity[1]);  // NB Apop
            writer.Double(out.cellularity[2]);  // NB Necro
            writer.Double(out.cellularity[4]);  // SC Apop
            writer.Double(out.cellularity[5]);  // SC Necro
            writer.EndArray();
            writer.Key("cell_count");
            writer.StartArray();
            // Primage expects array in different order to what we implemented
            // So remap it here
            assert(std::size(out.cell_count) == 6);
            writer.Int64(out.cell_count[0]);  // NB Living
            writer.Int64(out.cell_count[3]);  // SC Living
            writer.Int64(out.cell_count[1]);  // NB Apop
            writer.Int64(out.cell_count[2]);  // NB Necro
            writer.Int64(out.cell_count[4]);  // SC Apop
            writer.Int64(out.cell_count[5]);  // SC Necro
            writer.EndArray();
            writer.Key("cell_count_init");
            writer.StartArray();
            assert(std::size(out.cell_count_init) == 6);
            writer.Int64(out.cell_count_init[0]);  // NB Living
            writer.Int64(out.cell_count_init[3]);  // SC Living
            writer.Int64(out.cell_count_init[1]);  // NB Apop
            writer.Int64(out.cell_count_init[2]);  // NB Necro
            writer.Int64(out.cell_count_init[4]);  // SC Apop
            writer.Int64(out.cell_count_init[5]);  // SC Necro
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
