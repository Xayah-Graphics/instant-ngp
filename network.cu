#include "network.cuh"

namespace ngp::network::detail {

    void wait_stream_state_for_event(ngp::StreamState& stream_state, const cudaEvent_t event) {
        ngp::legacy::cuda_check(cudaStreamWaitEvent(stream_state.stream, event, 0));
    }

    void wait_stream_state_for_stream(ngp::StreamState& stream_state, const cudaStream_t stream) {
        ngp::legacy::cuda_check(cudaEventRecord(stream_state.event, stream));
        wait_stream_state_for_event(stream_state, stream_state.event);
    }

    void signal_stream_state(ngp::StreamState& stream_state, const cudaStream_t stream) {
        ngp::legacy::cuda_check(cudaEventRecord(stream_state.event, stream_state.stream));
        ngp::legacy::cuda_check(cudaStreamWaitEvent(stream, stream_state.event, 0));
    }

    std::unordered_map<cudaStream_t, std::stack<std::shared_ptr<ngp::StreamState>>>& stream_pools() {
        static auto* pools = new std::unordered_map<cudaStream_t, std::stack<std::shared_ptr<ngp::StreamState>>>{};
        return *pools;
    }

    void free_aux_stream_pool(const cudaStream_t parent_stream) {
        ngp::legacy::check_or_throw(parent_stream != nullptr);
        stream_pools().erase(parent_stream);
    }

    std::shared_ptr<ngp::StreamState> acquire_aux_stream(const cudaStream_t parent_stream) {
        ngp::legacy::check_or_throw(parent_stream != nullptr);
        auto& pool = stream_pools()[parent_stream];
        if (pool.empty()) pool.push(std::make_shared<ngp::StreamState>());

        auto result = pool.top();
        pool.pop();
        return result;
    }

    void release_aux_stream(const cudaStream_t parent_stream, std::shared_ptr<ngp::StreamState> aux_stream) {
        if (!stream_pools().contains(parent_stream)) throw std::runtime_error{"Attempted to return stream group to the wrong parent stream."};

        auto& pool = stream_pools()[parent_stream];
        pool.push(std::move(aux_stream));
    }

    SyncedStreamReservation::SyncedStreamReservation(const cudaStream_t stream, const std::size_t n_streams) : main_stream{stream} {
        if (n_streams == 0u) throw std::runtime_error{"SyncedStreamReservation: must request at least one stream"};
        if (n_streams == 1u) return;
        if (n_streams != 2u) throw std::runtime_error{"SyncedStreamReservation: this repository only supports a single auxiliary stream"};

        aux_stream = acquire_aux_stream(main_stream);
        wait_stream_state_for_stream(*aux_stream, main_stream);
    }

    SyncedStreamReservation::~SyncedStreamReservation() {
        if (!aux_stream) return;

        signal_stream_state(*aux_stream, main_stream);
        release_aux_stream(main_stream, std::move(aux_stream));
    }

    SyncedStreamReservation& SyncedStreamReservation::operator=(SyncedStreamReservation&& other) {
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
