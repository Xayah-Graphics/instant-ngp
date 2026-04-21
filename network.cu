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

    void wait_aux_stream_for_event(AuxStreamSlot& aux_stream, const cudaEvent_t event) {
        ngp::legacy::cuda_check(cudaStreamWaitEvent(aux_stream.stream, event, 0));
    }

    void wait_aux_stream_for_stream(AuxStreamSlot& aux_stream, const cudaStream_t stream) {
        ngp::legacy::cuda_check(cudaEventRecord(aux_stream.event, stream));
        wait_aux_stream_for_event(aux_stream, aux_stream.event);
    }

    void signal_aux_stream(AuxStreamSlot& aux_stream, const cudaStream_t stream) {
        ngp::legacy::cuda_check(cudaEventRecord(aux_stream.event, aux_stream.stream));
        ngp::legacy::cuda_check(cudaStreamWaitEvent(stream, aux_stream.event, 0));
    }

    std::unordered_map<cudaStream_t, std::stack<std::shared_ptr<AuxStreamSlot>>>& aux_stream_pools() {
        static auto* pools = new std::unordered_map<cudaStream_t, std::stack<std::shared_ptr<AuxStreamSlot>>>{};
        return *pools;
    }

    void free_aux_stream_pool(const cudaStream_t parent_stream) {
        ngp::legacy::check_or_throw(parent_stream != nullptr);
        aux_stream_pools().erase(parent_stream);
    }

    std::shared_ptr<AuxStreamSlot> acquire_aux_stream(const cudaStream_t parent_stream) {
        ngp::legacy::check_or_throw(parent_stream != nullptr);
        auto& pool = aux_stream_pools()[parent_stream];
        if (pool.empty()) pool.push(std::make_shared<AuxStreamSlot>());

        auto result = pool.top();
        pool.pop();
        return result;
    }

    void release_aux_stream(const cudaStream_t parent_stream, std::shared_ptr<AuxStreamSlot> aux_stream) {
        if (!aux_stream_pools().contains(parent_stream)) throw std::runtime_error{"Attempted to return stream group to the wrong parent stream."};

        auto& pool = aux_stream_pools()[parent_stream];
        pool.push(std::move(aux_stream));
    }

    SyncedStreamReservation::SyncedStreamReservation(const cudaStream_t stream, const std::size_t n_streams) : main_stream{stream} {
        if (n_streams == 0u) throw std::runtime_error{"SyncedStreamReservation: must request at least one stream"};
        if (n_streams == 1u) return;
        if (n_streams != 2u) throw std::runtime_error{"SyncedStreamReservation: this repository only supports a single auxiliary stream"};

        aux_stream_slot = acquire_aux_stream(main_stream);
        aux_stream      = aux_stream_slot->stream;
        wait_aux_stream_for_stream(*aux_stream_slot, main_stream);
    }

    SyncedStreamReservation::~SyncedStreamReservation() {
        if (!aux_stream_slot) return;

        signal_aux_stream(*aux_stream_slot, main_stream);
        release_aux_stream(main_stream, std::move(aux_stream_slot));
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
