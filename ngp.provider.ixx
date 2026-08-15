module;

#include <cuda_runtime_api.h>
#include <spectra/sdk/cuda_types.h>

export module ngp.provider;

import dataset.nerf_synthetic;
import ngp.train;
import spectra.sdk;
import spectra.sdk.cuda;
import std;

export namespace ngp {
    struct Settings {
        std::int64_t iterations_per_step{1};
    };

    struct Provider {
        Settings settings;

        static constexpr auto description = spectra::sdk::describe(
            "instant-ngp.field",
            spectra::sdk::parameter<"iterations_per_step", &Settings::iterations_per_step>(
                "Iterations Per Step",
                {},
                {.minimum = 1.0, .maximum = 64.0, .step = 1.0, .section = "Training"}
            ),
            spectra::sdk::hash_grid_radiance_field<"field">(),
            spectra::sdk::metric<"loss", float>("Loss", {}, "Training", true),
            spectra::sdk::metric<"iteration", std::uint64_t>("Iteration", {}, "Training")
        );

        Provider(Settings settings, const std::filesystem::path& assets);

        void setup(spectra::sdk::cuda::Setup& setup);
        void reset(std::uint64_t seed);
        void step(double seconds);
        void publish(spectra::sdk::cuda::Output& output);

    private:
        dataset::nerf_synthetic::Dataset dataset;
        std::unique_ptr<train::InstantNGP> neural_field;
        float loss{};
    };

    Provider::Provider(Settings source, const std::filesystem::path& assets) : settings(std::move(source)) {
        std::expected<dataset::nerf_synthetic::Dataset, std::string> loaded = dataset::nerf_synthetic::load(assets.parent_path() / "data/nerf-synthetic/lego", {.frame_sets = {"train"}, .scene_scale = 0.33F});
        if (!loaded) throw std::runtime_error(std::format("Failed to load Instant NGP dataset: {}", loaded.error()));
        dataset = std::move(*loaded);
    }

    void Provider::setup(spectra::sdk::cuda::Setup& setup) {
        setup.hash_grid_radiance_field<"field">();
    }

    void Provider::reset(const std::uint64_t seed) {
        neural_field = std::make_unique<train::InstantNGP>(dataset, train::TrainingStateRequest{.seed = seed});
        loss = 0.0F;
    }

    void Provider::step(const double) {
        const std::expected<train::OptimizationStats, std::string> optimized = neural_field->optimize({.frame_set = "train", .iterations = static_cast<std::int32_t>(settings.iterations_per_step)});
        if (!optimized) throw std::runtime_error(std::format("Instant NGP optimization failed: {}", optimized.error()));
        loss = optimized->loss;
    }

    void Provider::publish(spectra::sdk::cuda::Output& output) {
        const cudaStream_t stream{};
        spectra::sdk::cuda::Frame frame = output.begin(stream);
        spectra::sdk::cuda::HashGridRadianceField field = frame.hash_grid_radiance_field<"field">();
        const train::NeuralFieldParameters parameters = neural_field->neural_field_parameters();
        if (cudaMemcpyAsync(field.hash_grid.data(), parameters.hash_grid.data(), parameters.hash_grid.size_bytes(), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) throw std::runtime_error("Instant NGP hash grid publication failed");
        if (cudaMemcpyAsync(field.density_input.data(), parameters.density_input.data(), parameters.density_input.size_bytes(), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) throw std::runtime_error("Instant NGP density input publication failed");
        if (cudaMemcpyAsync(field.density_output.data(), parameters.density_output.data(), parameters.density_output.size_bytes(), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) throw std::runtime_error("Instant NGP density output publication failed");
        if (cudaMemcpyAsync(field.rgb_input.data(), parameters.rgb_input.data(), parameters.rgb_input.size_bytes(), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) throw std::runtime_error("Instant NGP RGB input publication failed");
        if (cudaMemcpyAsync(field.rgb_hidden.data(), parameters.rgb_hidden.data(), parameters.rgb_hidden.size_bytes(), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) throw std::runtime_error("Instant NGP RGB hidden publication failed");
        if (cudaMemcpyAsync(field.rgb_output.data(), parameters.rgb_output.data(), parameters.rgb_output.size_bytes(), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) throw std::runtime_error("Instant NGP RGB output publication failed");
        if (cudaMemcpyAsync(field.occupancy.data(), parameters.occupancy.data(), parameters.occupancy.size_bytes(), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) throw std::runtime_error("Instant NGP occupancy publication failed");
        frame.metric<"loss">().upload(loss);
        frame.metric<"iteration">().upload(static_cast<std::uint64_t>(neural_field->host.current_step));
        frame.commit();
    }
}
