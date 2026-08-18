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
            spectra::sdk::cameras<"training">(),
            spectra::sdk::hash_grid_radiance_field<"field">(),
            spectra::sdk::metric<"loss", float>("Loss", {}, "Training", true),
            spectra::sdk::metric<"psnr", float>("PSNR", "dB", "Training", true),
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
        float psnr{};
    };

    Provider::Provider(Settings source, const std::filesystem::path& assets) : settings(std::move(source)) {
        std::expected<dataset::nerf_synthetic::Dataset, std::string> loaded = dataset::nerf_synthetic::load(assets.parent_path().parent_path().parent_path() / "data/nerf-synthetic/lego", {.frame_sets = {"train"}, .scene_scale = 0.33F});
        if (!loaded) throw std::runtime_error(std::format("Failed to load Instant NGP dataset: {}", loaded.error()));
        dataset = std::move(*loaded);
    }

    void Provider::setup(spectra::sdk::cuda::Setup& setup) {
        const dataset::nerf_synthetic::FrameSet& training = dataset.frame_sets.front();
        const std::uint32_t width                          = training.frames.front().width;
        const std::uint32_t height                         = training.frames.front().height;
        std::vector<spectra::sdk::Camera> cameras{};
        cameras.reserve(training.frames.size());
        for (const dataset::nerf_synthetic::Frame& frame : training.frames) {
            cameras.emplace_back(spectra::sdk::Camera{
                .right     = {frame.camera[0], frame.camera[1], frame.camera[2]},
                .down      = {frame.camera[3], frame.camera[4], frame.camera[5]},
                .forward   = {frame.camera[6], frame.camera[7], frame.camera[8]},
                .position  = {frame.camera[9], frame.camera[10], frame.camera[11]},
                .focal     = {frame.focal_x, frame.focal_y},
                .principal = {frame.principal_x, frame.principal_y},
            });
        }
        spectra::sdk::cuda::CamerasSetup camera_output = setup.cameras<"training">(cameras, width, height);
        const std::size_t layer_byte_size              = static_cast<std::size_t>(width) * height * sizeof(spectra::sdk::Rgba8);
        for (std::size_t index = 0; index != training.frames.size(); ++index)
            if (cudaMemcpy(camera_output.images.data() + index * static_cast<std::size_t>(width) * height, training.frames[index].rgba.data(), layer_byte_size, cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("Instant NGP training Camera reference upload failed");
        setup.hash_grid_radiance_field<"field">();
    }

    void Provider::reset(const std::uint64_t seed) {
        neural_field = std::make_unique<train::InstantNGP>(dataset, train::TrainingStateRequest{.seed = seed});
        loss = 0.0F;
        psnr = std::numeric_limits<float>::quiet_NaN();
    }

    void Provider::step(const double) {
        const std::expected<train::OptimizationStats, std::string> optimized = neural_field->optimize({.frame_set = "train", .iterations = static_cast<std::int32_t>(settings.iterations_per_step)});
        if (!optimized) throw std::runtime_error(std::format("Instant NGP optimization failed: {}", optimized.error()));
        loss = optimized->loss;
        psnr = -10.0F * std::log10(loss);
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
        frame.metric<"psnr">().upload(psnr);
        frame.metric<"iteration">().upload(static_cast<std::uint64_t>(neural_field->host.current_step));
        frame.commit();
    }
}
