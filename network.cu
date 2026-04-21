#include "network.cuh"

namespace ngp::network::detail {

    struct AuxStreamSlot final {
        AuxStreamSlot() {
            ngp::legacy::cuda_check(cudaStreamCreate(&stream));
            ngp::legacy::cuda_check(cudaEventCreate(&event));
        }

        ~AuxStreamSlot() {
            if (event) cudaEventDestroy(event);
            if (stream) cudaStreamDestroy(stream);
        }

        AuxStreamSlot& operator=(const AuxStreamSlot&) = delete;
        AuxStreamSlot(const AuxStreamSlot&)            = delete;

        cudaStream_t stream = {};
        cudaEvent_t event   = {};
    };

    std::unordered_map<cudaStream_t, std::stack<std::unique_ptr<AuxStreamSlot, void (*)(AuxStreamSlot*)>>>& aux_stream_pools() {
        static auto* pools = new std::unordered_map<cudaStream_t, std::stack<std::unique_ptr<AuxStreamSlot, void (*)(AuxStreamSlot*)>>>{};
        return *pools;
    }

    void free_aux_stream_pool(const cudaStream_t parent_stream) {
        ngp::legacy::check_or_throw(parent_stream != nullptr);
        aux_stream_pools().erase(parent_stream);
    }

    SyncedStreamReservation::SyncedStreamReservation(const cudaStream_t stream, const std::size_t n_streams) : main_stream{stream} {
        if (n_streams == 0u) throw std::runtime_error{"SyncedStreamReservation: must request at least one stream"};
        if (n_streams == 1u) return;
        if (n_streams != 2u) throw std::runtime_error{"SyncedStreamReservation: this repository only supports a single auxiliary stream"};

        ngp::legacy::check_or_throw(main_stream != nullptr);
        auto& pool = aux_stream_pools()[main_stream];
        if (pool.empty()) pool.push(std::unique_ptr<AuxStreamSlot, void (*)(AuxStreamSlot*)>{new AuxStreamSlot{}, +[](AuxStreamSlot* aux_stream) { delete aux_stream; }});
        aux_stream_slot = std::move(pool.top());
        pool.pop();
        aux_stream      = aux_stream_slot->stream;
        ngp::legacy::cuda_check(cudaEventRecord(aux_stream_slot->event, main_stream));
        ngp::legacy::cuda_check(cudaStreamWaitEvent(aux_stream_slot->stream, aux_stream_slot->event, 0));
    }

    SyncedStreamReservation::~SyncedStreamReservation() {
        if (!aux_stream_slot) return;

        ngp::legacy::cuda_check(cudaEventRecord(aux_stream_slot->event, aux_stream_slot->stream));
        ngp::legacy::cuda_check(cudaStreamWaitEvent(main_stream, aux_stream_slot->event, 0));
        auto& pools = aux_stream_pools();
        if (!pools.contains(main_stream)) std::terminate();
        pools[main_stream].push(std::move(aux_stream_slot));
        aux_stream = nullptr;
    }

    SyncedStreamReservation& SyncedStreamReservation::operator=(SyncedStreamReservation&& other) {
        std::swap(aux_stream_slot, other.aux_stream_slot);
        std::swap(aux_stream, other.aux_stream);
        std::swap(main_stream, other.main_stream);
        return *this;
    }

    SyncedStreamReservation::SyncedStreamReservation(SyncedStreamReservation&& other) {
        *this = std::move(other);
    }

} // namespace ngp::network::detail

namespace ngp::encoding {}

namespace ngp::mlp {

    template class FullyFusedMLP<__half, ngp::density_network_width>;
#if NGP_RGB_NETWORK_WIDTH != NGP_DENSITY_NETWORK_WIDTH
    template class FullyFusedMLP<__half, ngp::rgb_network_width>;
#endif

} // namespace ngp::mlp

namespace ngp::optimizer {}

namespace ngp::network {}
