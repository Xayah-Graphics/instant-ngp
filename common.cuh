#ifndef NGP_LEGACY_CUH
#define NGP_LEGACY_CUH

#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <deque>
#include <functional>
#include <source_location>
#include <sstream>
#include <stdexcept>
#include <string>


namespace ngp::legacy {

    template <typename T>
    __host__ __device__ T next_multiple(T val, T divisor) {
        return ((val + divisor - 1) / divisor) * divisor;
    }

    template <typename T>
    __host__ __device__ T previous_multiple(T val, T divisor) {
        return (val / divisor) * divisor;
    }

    [[noreturn]] inline void throw_runtime_error(std::string_view message, const std::source_location& location = std::source_location::current()) {
        std::ostringstream stream;
        stream << location.file_name() << ':' << location.line() << ' ' << message;
        throw std::runtime_error{stream.str()};
    }

    inline void check_or_throw(bool condition, std::string_view message = "check failed", const std::source_location& location = std::source_location::current()) {
        if (!condition) {
            throw_runtime_error(message, location);
        }
    }

    inline void cu_check(CUresult result, const std::source_location& location = std::source_location::current()) {
        if (result == CUDA_SUCCESS) {
            return;
        }

        const char* message = nullptr;
        cuGetErrorName(result, &message);
        std::ostringstream stream;
        stream << "CUDA driver call failed: " << (message ? message : "unknown error");
        throw_runtime_error(stream.str(), location);
    }

    inline void cuda_check(cudaError_t result, const std::source_location& location = std::source_location::current()) {
        if (result == cudaSuccess) {
            return;
        }

        std::ostringstream stream;
        stream << "CUDA runtime call failed: " << cudaGetErrorString(result);
        throw_runtime_error(stream.str(), location);
    }


    struct GraphCaptureState {
        cudaGraph_t graph                  = nullptr;
        cudaGraphExec_t graph_instance     = nullptr;
        bool synchronize_when_capture_done = false;
    };


    inline std::deque<GraphCaptureState*>& current_graph_captures() {
        static thread_local std::deque<GraphCaptureState*> s_current_captures;
        return s_current_captures;
    }

    inline GraphCaptureState* current_graph_capture() {
        return current_graph_captures().empty() ? nullptr : current_graph_captures().front();
    }

    inline void reset_graph_capture(GraphCaptureState& graph_state) {
        if (graph_state.graph) {
            cuda_check(cudaGraphDestroy(graph_state.graph));
            graph_state.graph = nullptr;
        }

        if (graph_state.graph_instance) {
            cuda_check(cudaGraphExecDestroy(graph_state.graph_instance));
            graph_state.graph_instance = nullptr;
        }
    }

    inline void destroy_graph_capture(GraphCaptureState& graph_state) {
        try {
            reset_graph_capture(graph_state);
        } catch (const std::runtime_error& error) {
            if (std::string{error.what()}.find("driver shutting down") == std::string::npos) {
                std::fprintf(stderr, "Could not destroy cuda graph: %s\n", error.what());
            }
        }
    }

    inline void schedule_graph_capture_synchronize(GraphCaptureState& graph_state) {
        graph_state.synchronize_when_capture_done = true;
    }

    template <typename CaptureFn>
    void run_graph_capture(GraphCaptureState& graph_state, cudaStream_t stream, CaptureFn&& capture_fn) {
        if (stream == nullptr || stream == cudaStreamLegacy) {
            capture_fn();
            return;
        }

        cudaStreamCaptureStatus capture_status;
        cuda_check(cudaStreamIsCapturing(stream, &capture_status));
        if (capture_status != cudaStreamCaptureStatusNone) {
            capture_fn();
            return;
        }

        cudaError_t capture_result = cudaStreamIsCapturing(cudaStreamLegacy, &capture_status);
        if (capture_result == cudaErrorStreamCaptureImplicit) {
            capture_fn();
            return;
        }

        cuda_check(capture_result);
        if (capture_status != cudaStreamCaptureStatusNone) {
            capture_fn();
            return;
        }

        if (graph_state.graph) {
            cuda_check(cudaGraphDestroy(graph_state.graph));
            graph_state.graph = nullptr;
        }

        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
        current_graph_captures().push_back(&graph_state);
        try {
            capture_fn();
            cuda_check(cudaStreamEndCapture(stream, &graph_state.graph));

            if (current_graph_captures().back() != &graph_state) {
                throw std::runtime_error{"Graph capture must end in reverse order of creation."};
            }
            current_graph_captures().pop_back();
        } catch (...) {
            if (!current_graph_captures().empty() && current_graph_captures().back() == &graph_state) {
                current_graph_captures().pop_back();
            }

            cudaGraph_t aborted_graph      = nullptr;
            cudaError_t end_capture_result = cudaStreamEndCapture(stream, &aborted_graph);
            if (end_capture_result == cudaSuccess && aborted_graph) {
                cudaGraphDestroy(aborted_graph);
            } else {
                cudaGetLastError();
            }

            graph_state.graph = nullptr;
            throw;
        }

        if (graph_state.synchronize_when_capture_done) {
            cuda_check(cudaDeviceSynchronize());
            graph_state.synchronize_when_capture_done = false;
        }

        if (!graph_state.graph) {
            if (graph_state.graph_instance) {
                cuda_check(cudaGraphExecDestroy(graph_state.graph_instance));
            }

            graph_state.graph          = nullptr;
            graph_state.graph_instance = nullptr;
            return;
        }

        if (graph_state.graph_instance) {
            cudaGraphExecUpdateResultInfo update_result;
            cuda_check(cudaGraphExecUpdate(graph_state.graph_instance, graph_state.graph, &update_result));

            if (update_result.result != cudaGraphExecUpdateSuccess) {
                cuda_check(cudaGraphExecDestroy(graph_state.graph_instance));
                graph_state.graph_instance = nullptr;
            }
        }

        if (!graph_state.graph_instance) {
            cuda_check(cudaGraphInstantiate(&graph_state.graph_instance, graph_state.graph, nullptr, nullptr, 0));
        }

        cuda_check(cudaGraphLaunch(graph_state.graph_instance, stream));
    }


    inline std::atomic<size_t>& total_n_bytes_allocated() {
        static std::atomic<size_t> s_total_n_bytes_allocated{0};
        return s_total_n_bytes_allocated;
    }

    inline size_t align_to_cacheline(size_t bytes) {
        return next_multiple(bytes, static_cast<size_t>(128));
    }

    class ScopeGuard {
    public:
        explicit ScopeGuard(std::function<void()> callback) : m_callback{std::move(callback)} {}
        ScopeGuard& operator=(const ScopeGuard&) = delete;
        ScopeGuard(const ScopeGuard&)            = delete;
        ScopeGuard& operator=(ScopeGuard&& other) noexcept {
            std::swap(m_callback, other.m_callback);
            return *this;
        }
        ScopeGuard(ScopeGuard&& other) noexcept {
            *this = std::move(other);
        }
        ~ScopeGuard() {
            if (m_callback) {
                m_callback();
            }
        }

    private:
        std::function<void()> m_callback;
    };

    template <typename T>
    struct Interval {
        T start, end;

        __host__ __device__ bool operator<(const Interval& other) const {
            return end < other.end || (end == other.end && start < other.start);
        }

        __host__ __device__ [[nodiscard]] bool overlaps(const Interval& other) const {
            return !intersect(other).empty();
        }

        __host__ __device__ [[nodiscard]] Interval intersect(const Interval& other) const {
            return {std::max(start, other.start), std::min(end, other.end)};
        }

        __host__ __device__ [[nodiscard]] bool valid() const {
            return end >= start;
        }

        __host__ __device__ [[nodiscard]] bool empty() const {
            return end <= start;
        }

        __host__ __device__ [[nodiscard]] T size() const {
            return end - start;
        }
    };


    class GpuHeap {
    public:
        GpuHeap() {
            cuda_check(cudaGetDevice(&m_device));
            m_alignment = static_cast<size_t>(128);

            CUmemAllocationProp prop = {};
            prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id         = m_device;

            CUresult granularity_result = cuMemGetAllocationGranularity(&m_granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
            if (granularity_result == CUDA_ERROR_NOT_SUPPORTED) {
                m_granularity = 1;
            } else {
                cu_check(granularity_result);
            }

            size_t free_bytes;
            size_t total_bytes;
            cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
            m_max_size              = previous_multiple(total_bytes, m_granularity);
            m_global_free_intervals = {{0, m_max_size}};

            cu_check(cuMemAddressReserve(&m_base_address, m_max_size, 0, 0, 0));
        }

        GpuHeap(const GpuHeap&)            = delete;
        GpuHeap& operator=(const GpuHeap&) = delete;

        ~GpuHeap() {
            try {
                if (m_base_address) {
                    int previous_device;
                    cuda_check(cudaGetDevice(&previous_device));
                    cuda_check(cudaSetDevice(m_device));
                    ScopeGuard revert_device{[&]() { cuda_check(cudaSetDevice(previous_device)); }};

                    cuda_check(cudaDeviceSynchronize());
                    if (m_mapped_bytes) {
                        total_n_bytes_allocated() -= m_mapped_bytes;
                        cu_check(cuMemUnmap(m_base_address, m_mapped_bytes));
                    }
                    for (const auto& handle : m_handles) {
                        cu_check(cuMemRelease(handle));
                    }
                    cu_check(cuMemAddressFree(m_base_address, m_max_size));
                }
            } catch (const std::runtime_error& error) {
                if (std::string{error.what()}.find("driver shutting down") == std::string::npos) {
                    std::fprintf(stderr, "Could not free gpu heap: %s\n", error.what());
                }
            }
        }

        [[nodiscard]] uint8_t* data() const {
            return reinterpret_cast<uint8_t*>(m_base_address);
        }

        size_t allocate(size_t n_bytes, cudaStream_t stream) {
            if (n_bytes == 0) {
                return 0;
            }

            n_bytes = align_to_cacheline(n_bytes);

            if (stream && stream != cudaStreamLegacy) {
                auto& free_intervals             = m_stream_free_intervals[stream];
                Interval<size_t>* best_candidate = nullptr;
                for (auto& interval : free_intervals) {
                    if (interval.size() >= n_bytes && (!best_candidate || interval.size() < best_candidate->size())) {
                        best_candidate = &interval;
                    }
                }

                if (best_candidate) {
                    const size_t offset = best_candidate->start;
                    best_candidate->start += n_bytes;
                    if (best_candidate->start == best_candidate->end) {
                        std::erase_if(free_intervals, [](const Interval<size_t>& interval) { return interval.start == interval.end; });
                    }
                    return offset;
                }
            }

            Interval<size_t>* best_candidate = nullptr;
            for (auto& interval : m_global_free_intervals) {
                if (interval.size() >= n_bytes && (!best_candidate || interval.size() < best_candidate->size())) {
                    best_candidate = &interval;
                }
            }

            if (!best_candidate) {
                std::ostringstream basic_ostringstream;
                basic_ostringstream << "GpuHeap: failed to allocate " << n_bytes << " bytes from a " << m_max_size << "-byte heap.";
                throw std::runtime_error{basic_ostringstream.str()};
            }

            const size_t offset = best_candidate->start;
            best_candidate->start += n_bytes;
            if (best_candidate->start == best_candidate->end) {
                std::erase_if(m_global_free_intervals, [](const Interval<size_t>& interval) { return interval.start == interval.end; });
            }

            grow(offset + n_bytes);
            return offset;
        }

        void release(size_t offset, size_t n_bytes, cudaStream_t stream) {
            if (n_bytes == 0) {
                return;
            }

            n_bytes = align_to_cacheline(n_bytes);

            if (stream && stream != cudaStreamLegacy) {
                auto& free_intervals = m_stream_free_intervals[stream];
                auto pos             = std::upper_bound(free_intervals.begin(), free_intervals.end(), offset, [](size_t start, const Interval<size_t>& interval) { return start < interval.start; });
                free_intervals.insert(pos, {offset, offset + n_bytes});
                merge_adjacent(free_intervals);
                return;
            }

            cuda_check(cudaDeviceSynchronize());
            insert_global_free_interval({offset, offset + n_bytes});
        }

    private:
        void insert_global_free_interval(Interval<size_t> interval) {
            auto pos = std::upper_bound(m_global_free_intervals.begin(), m_global_free_intervals.end(), interval, [](const Interval<size_t>& a, const Interval<size_t>& b) { return a.start < b.start; });
            m_global_free_intervals.insert(pos, interval);
            merge_adjacent(m_global_free_intervals);
        }

        static void merge_adjacent(std::vector<Interval<size_t>>& intervals) {
            if (intervals.empty()) {
                return;
            }

            size_t j = 0;
            for (size_t i = 1; i < intervals.size(); ++i) {
                Interval<size_t>& prev = intervals[j];
                Interval<size_t>& cur  = intervals[i];
                if (prev.end == cur.start) {
                    prev.end = cur.end;
                } else {
                    ++j;
                    intervals[j] = cur;
                }
            }
            intervals.resize(j + 1);
        }

        void grow(size_t required_bytes) {
            if (required_bytes <= m_mapped_bytes) {
                return;
            }

            int current_device;
            cuda_check(cudaGetDevice(&current_device));
            if (current_device != m_device) {
                std::ostringstream stream;
                stream << "Attempted to use a GpuHeap of device " << m_device << " from the wrong device " << current_device << '.';
                throw std::runtime_error{stream.str()};
            }

            size_t n_bytes_to_allocate = required_bytes - m_mapped_bytes;
            n_bytes_to_allocate        = next_multiple(n_bytes_to_allocate, m_granularity);

            CUmemAllocationProp prop = {};
            prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id         = m_device;

            m_handles.emplace_back();
            cu_check(cuMemCreate(&m_handles.back(), n_bytes_to_allocate, &prop, 0));

            CUmemAccessDesc access_desc = {};
            access_desc.location.type   = CU_MEM_LOCATION_TYPE_DEVICE;
            access_desc.location.id     = m_device;
            access_desc.flags           = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

            cu_check(cuMemMap(m_base_address + m_mapped_bytes, n_bytes_to_allocate, 0, m_handles.back(), 0));
            cu_check(cuMemSetAccess(m_base_address + m_mapped_bytes, n_bytes_to_allocate, &access_desc, 1));
            m_mapped_bytes += n_bytes_to_allocate;
            total_n_bytes_allocated() += n_bytes_to_allocate;

            if (current_graph_capture()) {
                schedule_graph_capture_synchronize(*current_graph_capture());
            } else {
                cuda_check(cudaDeviceSynchronize());
            }
        }

        int m_device               = 0;
        CUdeviceptr m_base_address = {};
        size_t m_mapped_bytes      = 0;
        size_t m_alignment         = 0;
        size_t m_granularity       = 0;
        size_t m_max_size          = 0;
        std::vector<CUmemGenericAllocationHandle> m_handles;
        std::vector<Interval<size_t>> m_global_free_intervals;
        std::unordered_map<cudaStream_t, std::vector<Interval<size_t>>> m_stream_free_intervals;
    };
} // namespace ngp::legacy

#endif // NGP_LEGACY_CUH
