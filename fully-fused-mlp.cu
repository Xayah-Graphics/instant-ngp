#include "fully-fused-mlp.cuh"
#include <algorithm>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <mma.h>
#include <source_location>
#include <sstream>
#include <type_traits>

namespace ngp::mlp {

    inline void cutlass_check(const cutlass::Status result, const std::source_location& location = std::source_location::current()) {
        if (result == cutlass::Status::kSuccess) return;

        std::ostringstream stream;
        stream << "CUTLASS call failed: " << cutlassGetStatusString(result);
        legacy::throw_runtime_error(stream.str(), location);
    }

    template <legacy::MatrixLayout Layout>
    struct CutlassLayout final {
        typedef cutlass::layout::ColumnMajor type;
    };

    template <>
    struct CutlassLayout<legacy::RM> final {
        typedef cutlass::layout::RowMajor type;
    };

    template <>
    struct CutlassLayout<legacy::CM> final {
        typedef cutlass::layout::ColumnMajor type;
    };

    template <typename T>
    struct CutlassElementType final {
        typedef cutlass::half_t type;
    };

    template <>
    struct CutlassElementType<float> final {
        typedef float type;
    };

    template <typename ThreadBlock, typename Warp>
    struct LayerConfig {
        typedef ThreadBlock thread_block_shape;
        typedef Warp warp_shape;
    };

    struct FullLayerK : LayerConfig<cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<32, 32, 32>> {};
    struct LastLayerK : FullLayerK {};

    struct FullLayer : LayerConfig<cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>> {};
    struct LastLayer : FullLayer {};

    template <typename V>
    struct CutlassFragmentWrapper final {
        static constexpr std::uint32_t num_elements = V::kElements;
        V x                                         = {};
    };

    __device__ inline float logistic(const float x) {
        return 1.0f / (1.0f + expf(-x));
    }

    template <typename V>
    struct VectorFragment final {
        static constexpr std::uint32_t num_elements = V::size();
        V x                                         = {};
    };

    template <typename T>
    __device__ T relu(T value) {
        return static_cast<T>(cuda::std::max(static_cast<float>(value), 0.0f));
    }

    template <>
    inline __device__ half relu(half value) {
#if defined(__CUDA_ARCH__)
        return __hmax(value, static_cast<half>(0.0f));
#else
        return static_cast<half>(relu<float>(static_cast<float>(value)));
#endif
    }

    inline constexpr float k_act = 10.0f;

    template <typename T, typename Fragment, Activation activation>
        requires (activation == Activation::None)
    __device__ void warp_activation(const Fragment& frag, Fragment& result) {
        result = frag;
    }

    template <typename T, typename Fragment, Activation activation>
        requires (activation == Activation::ReLU)
    __device__ void warp_activation(const Fragment& frag, Fragment& result) {
        TCNN_PRAGMA_UNROLL
        for (int t = 0; t < static_cast<int>(result.num_elements); ++t) result.x[t] = relu(static_cast<T>(frag.x[t]));
    }

    template <typename T, typename Fragment, Activation activation>
        requires (activation == Activation::LeakyReLU)
    __device__ void warp_activation(const Fragment& frag, Fragment& result) {
        TCNN_PRAGMA_UNROLL
        for (int t = 0; t < static_cast<int>(result.num_elements); ++t) result.x[t] = frag.x[t] * static_cast<T>(static_cast<T>(frag.x[t]) > static_cast<T>(0.0f) ? 1.0f : 0.01f);
    }

    template <typename T, typename Fragment, Activation activation>
        requires (activation == Activation::Exponential)
    __device__ void warp_activation(const Fragment& frag, Fragment& result) {
        TCNN_PRAGMA_UNROLL
        for (int t = 0; t < static_cast<int>(result.num_elements); ++t) result.x[t] = static_cast<T>(expf(static_cast<float>(frag.x[t])));
    }

    template <typename T, typename Fragment, Activation activation>
        requires (activation == Activation::Sigmoid)
    __device__ void warp_activation(const Fragment& frag, Fragment& result) {
        TCNN_PRAGMA_UNROLL
        for (int t = 0; t < static_cast<int>(result.num_elements); ++t) result.x[t] = static_cast<T>(logistic(static_cast<float>(frag.x[t])));
    }

    template <typename T, typename Fragment, Activation activation>
        requires (activation == Activation::Squareplus)
    __device__ void warp_activation(const Fragment& frag, Fragment& result) {
        TCNN_PRAGMA_UNROLL
        for (int t = 0; t < static_cast<int>(result.num_elements); ++t) {
            const float x = static_cast<float>(frag.x[t]) * k_act;
            result.x[t]   = static_cast<T>(0.5f * (x + sqrtf(x * x + 4.0f)) / k_act);
        }
    }

    template <typename T, typename Fragment, Activation activation>
        requires (activation == Activation::Softplus)
    __device__ void warp_activation(const Fragment& frag, Fragment& result) {
        TCNN_PRAGMA_UNROLL
        for (int t = 0; t < static_cast<int>(result.num_elements); ++t) result.x[t] = static_cast<T>(logf(expf(static_cast<float>(frag.x[t]) * k_act) + 1.0f) / k_act);
    }

    template <typename T, typename Fragment, Activation activation>
        requires (activation == Activation::Tanh)
    __device__ void warp_activation(const Fragment& frag, Fragment& result) {
        TCNN_PRAGMA_UNROLL
        for (int t = 0; t < static_cast<int>(result.num_elements); ++t) result.x[t] = static_cast<T>(tanhf(static_cast<float>(frag.x[t])));
    }

    template <typename T, typename Fragment>
    __device__ void warp_activation(const Activation activation, const Fragment& frag, Fragment& result) {
        switch (activation) {
        case Activation::ReLU: warp_activation<T, Fragment, Activation::ReLU>(frag, result); return;
        case Activation::LeakyReLU: warp_activation<T, Fragment, Activation::LeakyReLU>(frag, result); return;
        case Activation::Exponential: warp_activation<T, Fragment, Activation::Exponential>(frag, result); return;
        case Activation::Sigmoid: warp_activation<T, Fragment, Activation::Sigmoid>(frag, result); return;
        case Activation::Squareplus: warp_activation<T, Fragment, Activation::Squareplus>(frag, result); return;
        case Activation::Softplus: warp_activation<T, Fragment, Activation::Softplus>(frag, result); return;
        case Activation::Tanh: warp_activation<T, Fragment, Activation::Tanh>(frag, result); return;
        case Activation::None: warp_activation<T, Fragment, Activation::None>(frag, result); return;
        default: return;
        }
    }

    template <typename T, typename Fragment>
    __device__ Fragment warp_activation(const Activation activation, const Fragment& frag) {
        Fragment result = {};
        warp_activation<T>(activation, frag, result);
        return result;
    }

    template <typename T, typename Fragment, typename ForwardFragment>
    __device__ void warp_activation_backward(const Activation activation, const Fragment& frag, const ForwardFragment& forward_frag, Fragment& result) {
        switch (activation) {
        case Activation::ReLU:
            TCNN_PRAGMA_UNROLL
            for (int t = 0; t < static_cast<int>(result.num_elements); ++t) result.x[t] = frag.x[t] * static_cast<T>(forward_frag.x[t] > static_cast<T>(0.0f));
            return;
        case Activation::LeakyReLU:
            TCNN_PRAGMA_UNROLL
            for (int t = 0; t < static_cast<int>(result.num_elements); ++t) result.x[t] = frag.x[t] * static_cast<T>(forward_frag.x[t] > static_cast<T>(0.0f) ? 1.0f : 0.01f);
            return;
        case Activation::Exponential:
            TCNN_PRAGMA_UNROLL
            for (int t = 0; t < static_cast<int>(result.num_elements); ++t) result.x[t] = frag.x[t] * forward_frag.x[t];
            return;
        case Activation::Sigmoid:
            TCNN_PRAGMA_UNROLL
            for (int t = 0; t < static_cast<int>(result.num_elements); ++t) result.x[t] = frag.x[t] * static_cast<T>(forward_frag.x[t] * static_cast<T>(1.0f - static_cast<float>(forward_frag.x[t])));
            return;
        case Activation::Squareplus:
            TCNN_PRAGMA_UNROLL
            for (int t = 0; t < static_cast<int>(result.num_elements); ++t) {
                const float y = static_cast<float>(forward_frag.x[t]) * k_act;
                result.x[t]   = frag.x[t] * static_cast<T>(y * y / (y * y + 1.0f));
            }
            return;
        case Activation::Softplus:
            TCNN_PRAGMA_UNROLL
            for (int t = 0; t < static_cast<int>(result.num_elements); ++t) result.x[t] = frag.x[t] * static_cast<T>(1.0f - expf(-static_cast<float>(forward_frag.x[t]) * k_act));
            return;
        case Activation::Tanh:
            TCNN_PRAGMA_UNROLL
            for (int t = 0; t < static_cast<int>(result.num_elements); ++t) result.x[t] = frag.x[t] * static_cast<T>(1.0f - (static_cast<float>(forward_frag.x[t]) * static_cast<float>(forward_frag.x[t])));
            return;
        case Activation::None: result = frag; return;
        default: return;
        }
    }

    template <typename T, typename Fragment, typename ForwardFragment>
    __device__ Fragment warp_activation_backward(const Activation activation, const Fragment& frag, const ForwardFragment& forward_frag) {
        Fragment result = {};
        warp_activation_backward<T>(activation, frag, forward_frag, result);
        return result;
    }

    template <typename T, std::uint32_t N = 1u>
    __global__ void kernel_activation_backward_output(const std::uint32_t num_elements, const Activation activation, const T* __restrict__ output_values, const T* gradients_out, T* gradients_in) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= num_elements) return;

        const auto frag_forward_out = reinterpret_cast<const VectorFragment<legacy::math::tvec<T, N, sizeof(T)>>*>(output_values)[i];
        auto frag                   = reinterpret_cast<const VectorFragment<legacy::math::tvec<T, N, sizeof(T)>>*>(gradients_out)[i];
        warp_activation_backward<T>(activation, frag, frag_forward_out, frag);
        reinterpret_cast<VectorFragment<legacy::math::tvec<T, N, sizeof(T)>>*>(gradients_in)[i] = frag;
    }

    template <typename ElementOutput_, int Count, typename ElementAccumulator_ = ElementOutput_, typename ElementCompute_ = ElementOutput_, cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest>
    class ActivationEpilogue {
    public:
        typedef ElementOutput_ ElementOutput;
        typedef ElementAccumulator_ ElementAccumulator;
        typedef ElementCompute_ ElementCompute;

        static constexpr int kCount = Count;

        typedef cutlass::Array<ElementOutput, kCount> FragmentOutput;
        typedef cutlass::Array<ElementAccumulator, kCount> FragmentAccumulator;
        typedef cutlass::Array<ElementCompute, kCount> ComputeFragment;

        static constexpr cutlass::FloatRoundStyle kRound = Round;

        struct Params {
            Activation activation;
        };

        CUTLASS_HOST_DEVICE
        explicit ActivationEpilogue(const Params& params) : m_activation{params.activation} {}

        CUTLASS_HOST_DEVICE
        bool is_source_needed() const {
            return false;
        }

        CUTLASS_HOST_DEVICE
        void set_k_partition(int, int) {}

        CUTLASS_HOST_DEVICE
        FragmentOutput operator()(const FragmentAccumulator& accumulator) const {
            cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

            CutlassFragmentWrapper<ComputeFragment> intermediate{accumulator_converter(accumulator)};
            intermediate = warp_activation<ElementCompute>(m_activation, intermediate);

            cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;
            return destination_converter(intermediate.x);
        }

        CUTLASS_HOST_DEVICE
        FragmentOutput operator()(const FragmentAccumulator& accumulator, const FragmentOutput&) const {
            return operator()(accumulator);
        }

    private:
        Activation m_activation;
    };

    template <typename ElementOutput_, int Count, typename ElementAccumulator_ = ElementOutput_, typename ElementCompute_ = ElementOutput_, cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest>
    class ActivationTransferEpilogue {
    public:
        typedef ElementOutput_ ElementOutput;
        typedef ElementAccumulator_ ElementAccumulator;
        typedef ElementCompute_ ElementCompute;

        static constexpr int kCount = Count;

        typedef cutlass::Array<ElementOutput, kCount> FragmentOutput;
        typedef cutlass::Array<ElementAccumulator, kCount> FragmentAccumulator;
        typedef cutlass::Array<ElementCompute, kCount> ComputeFragment;

        static constexpr cutlass::FloatRoundStyle kRound = Round;

        struct Params {
            Activation activation;
        };

        CUTLASS_HOST_DEVICE
        explicit ActivationTransferEpilogue(const Params& params) : m_activation{params.activation} {}

        CUTLASS_HOST_DEVICE
        bool is_source_needed() const {
            return true;
        }

        CUTLASS_HOST_DEVICE
        void set_k_partition(int, int) {}

        CUTLASS_HOST_DEVICE
        FragmentOutput operator()(const FragmentAccumulator& accumulator, const FragmentOutput& source) const {
            cutlass::NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
            cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

            CutlassFragmentWrapper<ComputeFragment> converted_source{source_converter(source)};
            CutlassFragmentWrapper<ComputeFragment> intermediate{accumulator_converter(accumulator)};
            intermediate = warp_activation_backward<ElementCompute>(m_activation, intermediate, converted_source);

            cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;
            return destination_converter(intermediate.x);
        }

        CUTLASS_HOST_DEVICE
        FragmentOutput operator()(const FragmentAccumulator& accumulator) const {
            cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;
            ComputeFragment converted_accumulator = accumulator_converter(accumulator);

            cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;
            return destination_converter(converted_accumulator);
        }

    private:
        Activation m_activation;
    };

    template <typename T>
    inline constexpr int n_vectorized_elements = 128 / cutlass::sizeof_bits<T>::value;

    template <class Gemm>
    void fc_multiply_impl(cudaStream_t stream, const typename Gemm::Arguments& args) {
        const std::size_t workspace_size = Gemm::get_workspace_size(args);
        Gemm gemm_op;

        auto workspace         = legacy::GpuAllocation{workspace_size, stream};
        cutlass::Status status = gemm_op.initialize(args, workspace.data(), stream);
        cutlass_check(status);

        status = gemm_op(stream);
        cutlass_check(status);
    }

    template <class Gemm>
    void fc_multiply_split_k_impl(cudaStream_t stream, const typename Gemm::Arguments& args) {
        const std::size_t workspace_size = Gemm::get_workspace_size(args);
        Gemm gemm_op;

        auto workspace         = legacy::GpuAllocation{workspace_size, stream};
        cutlass::Status status = gemm_op.initialize(args, workspace.data());
        cutlass_check(status);

        status = gemm_op(stream);
        cutlass_check(status);
    }

    template <typename Config, typename TypeA, legacy::MatrixLayout LayoutA, typename TypeB, legacy::MatrixLayout LayoutB, typename TypeC, legacy::MatrixLayout LayoutC, typename TypeD, legacy::MatrixLayout LayoutD>
    void fc_multiply(cudaStream_t stream, const legacy::GPUMatrix<TypeA, LayoutA>& A, const legacy::GPUMatrix<TypeB, LayoutB>& B, const legacy::GPUMatrix<TypeC, LayoutC>& C, const legacy::GPUMatrix<TypeD, LayoutD>& D, Activation act = Activation::None, bool transfer = false) {
        static_assert(std::is_same_v<TypeA, TypeB>, "Type of matrix A and B must be equal");
        static_assert(std::is_same_v<TypeC, TypeD>, "Type of matrix C and D must be equal");
        static_assert(std::is_same_v<typename CutlassLayout<LayoutC>::type, typename CutlassLayout<LayoutD>::type>, "Layout of matrix C and D must be equal");

        if (A.n() != B.m()) throw std::runtime_error{"Matrices A and B can not be multiplied together"};

        const int M = static_cast<int>(A.m());
        const int K = static_cast<int>(A.n());
        const int N = static_cast<int>(B.n());

        if (C.m() != static_cast<std::uint32_t>(M) || C.n() != static_cast<std::uint32_t>(N)) {
            std::ostringstream stream_message;
            stream_message << "Matrix C has incorrect size " << C.m() << 'x' << C.n() << " != " << M << 'x' << N;
            throw std::runtime_error{stream_message.str()};
        }

        if (D.m() != static_cast<std::uint32_t>(M) || D.n() != static_cast<std::uint32_t>(N)) {
            std::ostringstream stream_message;
            stream_message << "Matrix D has incorrect size " << D.m() << 'x' << D.n() << " != " << M << 'x' << N;
            throw std::runtime_error{stream_message.str()};
        }

        typedef typename CutlassElementType<TypeA>::type MatmulTypeCompute;
        typedef typename CutlassElementType<TypeC>::type MatmulTypeAccumulator;

        if (transfer) {
            typedef cutlass::gemm::device::Gemm<MatmulTypeCompute, typename CutlassLayout<LayoutA>::type, MatmulTypeCompute, typename CutlassLayout<LayoutB>::type, MatmulTypeAccumulator, typename CutlassLayout<LayoutC>::type, cutlass::half_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, typename Config::thread_block_shape, typename Config::warp_shape, cutlass::gemm::GemmShape<16, 8, 8>,
                ActivationTransferEpilogue<MatmulTypeAccumulator, n_vectorized_elements<MatmulTypeAccumulator>, cutlass::half_t, cutlass::half_t>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>
                Gemm;

            typename Gemm::Arguments arguments{{M, N, K}, {reinterpret_cast<MatmulTypeCompute*>(A.data()), static_cast<int>(A.stride())}, {reinterpret_cast<MatmulTypeCompute*>(B.data()), static_cast<int>(B.stride())}, {reinterpret_cast<MatmulTypeAccumulator*>(C.data()), static_cast<int>(C.stride())}, {reinterpret_cast<MatmulTypeAccumulator*>(D.data()), static_cast<int>(D.stride())}, {act}, 1};
            fc_multiply_impl<Gemm>(stream, arguments);
        } else {
            typedef cutlass::gemm::device::Gemm<MatmulTypeCompute, typename CutlassLayout<LayoutA>::type, MatmulTypeCompute, typename CutlassLayout<LayoutB>::type, MatmulTypeAccumulator, typename CutlassLayout<LayoutC>::type, cutlass::half_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, typename Config::thread_block_shape, typename Config::warp_shape, cutlass::gemm::GemmShape<16, 8, 8>,
                ActivationEpilogue<MatmulTypeAccumulator, n_vectorized_elements<MatmulTypeAccumulator>, cutlass::half_t, cutlass::half_t>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>
                Gemm;

            typename Gemm::Arguments arguments{{M, N, K}, {reinterpret_cast<MatmulTypeCompute*>(A.data()), static_cast<int>(A.stride())}, {reinterpret_cast<MatmulTypeCompute*>(B.data()), static_cast<int>(B.stride())}, {reinterpret_cast<MatmulTypeAccumulator*>(C.data()), static_cast<int>(C.stride())}, {reinterpret_cast<MatmulTypeAccumulator*>(D.data()), static_cast<int>(D.stride())}, {act}, 1};
            fc_multiply_impl<Gemm>(stream, arguments);
        }
    }

    template <typename Config, typename TypeA, legacy::MatrixLayout LayoutA, typename TypeB, legacy::MatrixLayout LayoutB, typename TypeC, typename TypeD>
    void fc_multiply(cudaStream_t stream, const legacy::GPUMatrix<TypeA, LayoutA>& A, const legacy::GPUMatrix<TypeB, LayoutB>& B, const legacy::GPUMatrix<TypeC, legacy::MatrixLayout::Dynamic>& C, const legacy::GPUMatrix<TypeD, legacy::MatrixLayout::Dynamic>& D, Activation act = Activation::None, bool transfer = false) {
        if (C.layout() != D.layout()) throw std::runtime_error{"fc_multiply: layout of dynamic GPUMatrix C and D must be equal"};
        if (D.layout() == legacy::CM)
            fc_multiply<Config>(stream, A, B, C.cm(), D.cm(), act, transfer);
        else
            fc_multiply<Config>(stream, A, B, C.rm(), D.rm(), act, transfer);
    }

    template <typename Config, typename TypeA, legacy::MatrixLayout LayoutA, typename TypeB, typename TypeC, typename TypeD>
    void fc_multiply(cudaStream_t stream, const legacy::GPUMatrix<TypeA, LayoutA>& A, const legacy::GPUMatrix<TypeB, legacy::MatrixLayout::Dynamic>& B, const legacy::GPUMatrix<TypeC, legacy::MatrixLayout::Dynamic>& C, const legacy::GPUMatrix<TypeD, legacy::MatrixLayout::Dynamic>& D, Activation act = Activation::None, bool transfer = false) {
        if (B.layout() == legacy::CM)
            fc_multiply<Config>(stream, A, B.cm(), C, D, act, transfer);
        else
            fc_multiply<Config>(stream, A, B.rm(), C, D, act, transfer);
    }

    template <typename Config, typename TypeA, legacy::MatrixLayout LayoutA, typename TypeB, typename TypeD>
    void fc_multiply(cudaStream_t stream, const legacy::GPUMatrix<TypeA, LayoutA>& A, const legacy::GPUMatrix<TypeB, legacy::MatrixLayout::Dynamic>& B, const legacy::GPUMatrix<TypeD, legacy::MatrixLayout::Dynamic>& D, Activation act = Activation::None) {
        fc_multiply<Config>(stream, A, B, D, D, act);
    }

    template <typename Config, typename TypeA, legacy::MatrixLayout LayoutA, typename TypeB, legacy::MatrixLayout LayoutB, typename TypeC, legacy::MatrixLayout LayoutC, typename TypeD, legacy::MatrixLayout LayoutD>
    void fc_multiply_split_k(cudaStream_t stream, const legacy::GPUMatrix<TypeA, LayoutA>& A, const legacy::GPUMatrix<TypeB, LayoutB>& B, const legacy::GPUMatrix<TypeC, LayoutC>& C, const legacy::GPUMatrix<TypeD, LayoutD>& D, std::uint32_t split_k_slices = 1u, float beta = 0.0f) {
        static_assert(std::is_same_v<TypeA, TypeB>, "Type of matrix A and B must be equal");
        static_assert(std::is_same_v<TypeC, TypeD>, "Type of matrix C and D must be equal");
        static_assert(std::is_same_v<typename CutlassLayout<LayoutC>::type, typename CutlassLayout<LayoutD>::type>, "Layout of matrix C and D must be equal");

        if (A.n() != B.m()) throw std::runtime_error{"Matrices A and B can not be multiplied together"};

        const int M = static_cast<int>(A.m());
        const int K = static_cast<int>(A.n());
        const int N = static_cast<int>(B.n());

        if (C.m() != static_cast<std::uint32_t>(M) || C.n() != static_cast<std::uint32_t>(N)) {
            std::ostringstream stream_message;
            stream_message << "Matrix C has incorrect size " << C.m() << 'x' << C.n() << " != " << M << 'x' << N;
            throw std::runtime_error{stream_message.str()};
        }

        if (D.m() != static_cast<std::uint32_t>(M) || D.n() != static_cast<std::uint32_t>(N)) {
            std::ostringstream stream_message;
            stream_message << "Matrix D has incorrect size " << D.m() << 'x' << D.n() << " != " << M << 'x' << N;
            throw std::runtime_error{stream_message.str()};
        }

        typedef typename CutlassElementType<TypeA>::type MatmulTypeCompute;
        typedef typename CutlassElementType<TypeC>::type MatmulTypeAccumulator;
        typedef cutlass::gemm::device::GemmSplitKParallel<MatmulTypeCompute, typename CutlassLayout<LayoutA>::type, MatmulTypeCompute, typename CutlassLayout<LayoutB>::type, MatmulTypeAccumulator, typename CutlassLayout<LayoutC>::type, cutlass::half_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, typename Config::thread_block_shape, typename Config::warp_shape, cutlass::gemm::GemmShape<16, 8, 8>,
            cutlass::epilogue::thread::LinearCombination<MatmulTypeAccumulator, n_vectorized_elements<MatmulTypeAccumulator>, cutlass::half_t, cutlass::half_t>>
            Gemm;

        typename Gemm::Arguments arguments{
            {M, N, K}, {reinterpret_cast<MatmulTypeCompute*>(A.data()), static_cast<int>(A.stride())}, {reinterpret_cast<MatmulTypeCompute*>(B.data()), static_cast<int>(B.stride())}, {reinterpret_cast<MatmulTypeAccumulator*>(C.data()), static_cast<int>(C.stride())}, {reinterpret_cast<MatmulTypeAccumulator*>(D.data()), static_cast<int>(D.stride())}, {static_cast<cutlass::half_t>(1.0f), static_cast<cutlass::half_t>(beta)}, static_cast<int>(split_k_slices)};
        fc_multiply_split_k_impl<Gemm>(stream, arguments);
    }

    template <typename Config, typename TypeA, legacy::MatrixLayout LayoutA, typename TypeB, legacy::MatrixLayout LayoutB, typename TypeC, typename TypeD>
    void fc_multiply_split_k(cudaStream_t stream, const legacy::GPUMatrix<TypeA, LayoutA>& A, const legacy::GPUMatrix<TypeB, LayoutB>& B, const legacy::GPUMatrix<TypeC, legacy::MatrixLayout::Dynamic>& C, const legacy::GPUMatrix<TypeD, legacy::MatrixLayout::Dynamic>& D, std::uint32_t split_k_slices = 1u, float beta = 0.0f) {
        if (C.layout() != D.layout()) throw std::runtime_error{"fc_multiply: layout of dynamic GPUMatrix C and D must be equal"};
        if (D.layout() == legacy::CM)
            fc_multiply_split_k<Config>(stream, A, B, C.cm(), D.cm(), split_k_slices, beta);
        else
            fc_multiply_split_k<Config>(stream, A, B, C.rm(), D.rm(), split_k_slices, beta);
    }

    template <typename Config, typename TypeA, legacy::MatrixLayout LayoutA, typename TypeB, typename TypeC, typename TypeD>
    void fc_multiply_split_k(cudaStream_t stream, const legacy::GPUMatrix<TypeA, LayoutA>& A, const legacy::GPUMatrix<TypeB, legacy::MatrixLayout::Dynamic>& B, const legacy::GPUMatrix<TypeC, legacy::MatrixLayout::Dynamic>& C, const legacy::GPUMatrix<TypeD, legacy::MatrixLayout::Dynamic>& D, std::uint32_t split_k_slices = 1u, float beta = 0.0f) {
        if (B.layout() == legacy::CM)
            fc_multiply_split_k<Config>(stream, A, B.cm(), C, D, split_k_slices, beta);
        else
            fc_multiply_split_k<Config>(stream, A, B.rm(), C, D, split_k_slices, beta);
    }

    template <typename Config, typename TypeA, typename TypeB, typename TypeC, typename TypeD>
    void fc_multiply_split_k(cudaStream_t stream, const legacy::GPUMatrix<TypeA, legacy::MatrixLayout::Dynamic>& A, const legacy::GPUMatrix<TypeB, legacy::MatrixLayout::Dynamic>& B, const legacy::GPUMatrix<TypeC, legacy::MatrixLayout::Dynamic>& C, const legacy::GPUMatrix<TypeD, legacy::MatrixLayout::Dynamic>& D, std::uint32_t split_k_slices = 1u, float beta = 0.0f) {
        if (A.layout() == legacy::CM)
            fc_multiply_split_k<Config>(stream, A.cm(), B, C, D, split_k_slices, beta);
        else
            fc_multiply_split_k<Config>(stream, A.rm(), B, C, D, split_k_slices, beta);
    }

    template <typename Config, typename TypeA, typename TypeB, typename TypeD>
    void fc_multiply_split_k(cudaStream_t stream, const legacy::GPUMatrix<TypeA, legacy::MatrixLayout::Dynamic>& A, const legacy::GPUMatrix<TypeB, legacy::MatrixLayout::Dynamic>& B, const legacy::GPUMatrix<TypeD, legacy::MatrixLayout::Dynamic>& D, std::uint32_t split_k_slices, float beta) {
        fc_multiply_split_k<Config>(stream, A, B, D, D, split_k_slices, beta);
    }

    template <typename T>
    void activation_backward_output_gpu(cudaStream_t stream, const std::uint32_t num_elements, const Activation act, const T* __restrict__ output_values, const T* gradients_out, T* gradients_in) {
        static constexpr std::uint32_t activation_vector_size = 16u / sizeof(T);
        if (num_elements % activation_vector_size != 0u) {
            std::ostringstream stream_message;
            stream_message << "activation_backward_output_gpu: number of elements must be a multiple of " << activation_vector_size;
            throw std::runtime_error{stream_message.str()};
        }

        if (act == Activation::None && gradients_out == gradients_in) return;

        const std::uint32_t vector_count = num_elements / activation_vector_size;
        if (vector_count > 0u) {
            const std::uint32_t blocks = (vector_count + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
            kernel_activation_backward_output<T, activation_vector_size><<<blocks, network::detail::n_threads_linear, 0, stream>>>(vector_count, act, output_values, gradients_out, gradients_in);
        }
    }

    inline void check_shmem_error(const cudaError_t error) {
        if (error == cudaSuccess) return;
        throw std::runtime_error{"FullyFusedMLP: insufficient shared memory available on the GPU. Reduce the selected compile-time network width to fit the device."};
    }

    template <std::uint32_t WIDTH, std::uint32_t N_ITERS, typename OUT_T, bool BACKWARD = false>
    __device__ void threadblock_layer(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out_intermediate_threadblock_this_layer, const OUT_T* __restrict__ activation_aux = nullptr) {
        constexpr std::uint32_t SKEW     = WIDTH % 16u == 0u ? 8u : 0u;
        constexpr std::uint32_t N_BLOCKS = WIDTH / 16u;

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> act_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, std::conditional_t<BACKWARD, nvcuda::wmma::row_major, nvcuda::wmma::col_major>> weights_frag[N_BLOCKS];
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, OUT_T> result_frag[N_ITERS];

        const std::uint32_t li = threadIdx.x;
        const std::uint32_t wi = threadIdx.y;

        const std::uint32_t lane_offset = (8u * li) % WIDTH;
        const std::uint32_t row         = (8u * li + wi * 8u * 32u) / WIDTH;
        const std::uint32_t weights_col = 16u * wi;

        __syncthreads();

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t i = 0u; i < N_BLOCKS; ++i) {
            if constexpr (BACKWARD)
                nvcuda::wmma::load_matrix_sync(weights_frag[i], weights_this_layer + 16u * i * WIDTH + weights_col, WIDTH);
            else
                nvcuda::wmma::load_matrix_sync(weights_frag[i], weights_this_layer + 16u * i + weights_col * WIDTH, WIDTH);
        }

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t l = 0u; l < N_ITERS; ++l) {
            nvcuda::wmma::fill_fragment(result_frag[l], 0.0f);

            TCNN_PRAGMA_UNROLL
            for (std::uint32_t i = 0u; i < N_BLOCKS; ++i) {
                nvcuda::wmma::load_matrix_sync(act_frag, act_shmem + 16u * i + (16u * l) * (WIDTH + SKEW), WIDTH + SKEW);
                nvcuda::wmma::mma_sync(result_frag[l], act_frag, weights_frag[i], result_frag[l]);
            }

            if constexpr (BACKWARD) {
                nvcuda::wmma::load_matrix_sync(act_frag, activation_aux + weights_col + l * 16u * WIDTH, WIDTH);
                warp_activation_backward<__half>(activation, result_frag[l], act_frag, result_frag[l]);
            } else {
                warp_activation<__half>(activation, result_frag[l], result_frag[l]);
            }
        }

        __syncthreads();

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t l = 0u; l < N_ITERS; ++l) nvcuda::wmma::store_matrix_sync(act_shmem + weights_col + l * 16u * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, nvcuda::wmma::mem_row_major);

        if (out_intermediate_threadblock_this_layer != nullptr) {
            __syncthreads();

            TCNN_PRAGMA_UNROLL
            for (std::uint32_t l = 0u; l < N_ITERS; ++l) *reinterpret_cast<int4*>(&out_intermediate_threadblock_this_layer[lane_offset + (row + 16u * l) * WIDTH]) = *reinterpret_cast<int4*>(&act_shmem[lane_offset + (row + 16u * l) * (WIDTH + SKEW)]);
        }
    }

    template <std::uint32_t WIDTH, std::uint32_t N_ITERS>
    __device__ void threadblock_load_input_static(__half* __restrict__ act_shmem, const __half* __restrict__ input_threadblock) {
        constexpr std::uint32_t SKEW = WIDTH % 16u == 0u ? 8u : 0u;

        const std::uint32_t li = threadIdx.x;
        const std::uint32_t wi = threadIdx.y;

        const std::uint32_t lane_offset = (8u * li) % WIDTH;
        const std::uint32_t row         = (8u * li + wi * 8u * 32u) / WIDTH;

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t i = 0u; i < N_ITERS; ++i) *reinterpret_cast<int4*>(&act_shmem[lane_offset + (row + 16u * i) * (WIDTH + SKEW)]) = *reinterpret_cast<const int4*>(&input_threadblock[lane_offset + (row + 16u * i) * WIDTH]);
    }

    template <std::uint32_t WIDTH, std::uint32_t N_ITERS, Activation ACTIVATION, typename OUTPUT_LAYOUT>
    __global__ void kernel_mlp_fused_backward(const __half* __restrict__ dL_doutput, const __half* __restrict__ weights, __half* __restrict__ out_intermediate, const __half* __restrict__ forward, __half* __restrict__ dL_dinput, const __half* __restrict__ weights_first_layer, const std::uint32_t output_stride, const std::uint32_t batch_size, const std::uint32_t out_width, const std::uint32_t n_hidden_matmuls) {
        constexpr std::uint32_t SKEW = WIDTH % 16u == 0u ? 8u : 0u;

        const std::uint32_t wi            = threadIdx.y;
        const std::uint32_t bi            = blockIdx.x;
        const std::uint32_t elem_idx_base = 16u * bi * N_ITERS;
        const std::uint32_t elem_idx      = elem_idx_base;

        extern __shared__ __half shmem[];
        __half* act_shmem = shmem;

        const std::uint32_t weights_stride = WIDTH * WIDTH;
        const std::uint32_t layer_stride   = WIDTH * batch_size;

        if (out_width <= 16u) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, OUTPUT_LAYOUT> act_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> weights_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> result_frag[N_ITERS];

            const std::uint32_t weights_col = 16u * wi;
            nvcuda::wmma::load_matrix_sync(weights_frag, weights + weights_stride * n_hidden_matmuls + weights_col, WIDTH);

            TCNN_PRAGMA_UNROLL
            for (std::uint32_t l = 0u; l < N_ITERS; ++l) {
                nvcuda::wmma::fill_fragment(result_frag[l], 0.0f);

                if constexpr (std::is_same_v<OUTPUT_LAYOUT, nvcuda::wmma::row_major>)
                    nvcuda::wmma::load_matrix_sync(act_frag, dL_doutput + (elem_idx + 16u * l) * output_stride, output_stride);
                else
                    nvcuda::wmma::load_matrix_sync(act_frag, dL_doutput + (elem_idx + 16u * l), output_stride);

                nvcuda::wmma::mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);

                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> forward_frag;
                nvcuda::wmma::load_matrix_sync(forward_frag, forward + layer_stride * n_hidden_matmuls + weights_col + (elem_idx + l * 16u) * WIDTH, WIDTH);
                warp_activation_backward<__half>(ACTIVATION, result_frag[l], forward_frag, result_frag[l]);
            }

            __syncthreads();

            TCNN_PRAGMA_UNROLL
            for (std::uint32_t l = 0u; l < N_ITERS; ++l) nvcuda::wmma::store_matrix_sync(act_shmem + weights_col + (16u * l) * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, nvcuda::wmma::mem_row_major);

            __syncthreads();

            const std::uint32_t li          = threadIdx.x;
            const std::uint32_t lane_offset = (8u * li) % WIDTH;
            const std::uint32_t row         = (8u * li + wi * 8u * 32u) / WIDTH;

            TCNN_PRAGMA_UNROLL
            for (std::uint32_t i = 0u; i < N_ITERS; ++i) *reinterpret_cast<int4*>(&out_intermediate[lane_offset + (row + elem_idx + i * 16u) * WIDTH]) = *reinterpret_cast<int4*>(&act_shmem[lane_offset + (row + 16u * i) * (WIDTH + SKEW)]);
        } else {
            threadblock_load_input_static<WIDTH, N_ITERS>(act_shmem, out_intermediate + elem_idx * WIDTH);
        }

        for (std::uint32_t k = 0u; k < n_hidden_matmuls; ++k) threadblock_layer<WIDTH, N_ITERS, __half, true>(ACTIVATION, act_shmem, weights + weights_stride * (n_hidden_matmuls - k - 1u), out_intermediate + layer_stride * (k + 1u) + elem_idx_base * WIDTH, forward + layer_stride * (n_hidden_matmuls - k - 1u) + elem_idx_base * WIDTH);

        if (dL_dinput != nullptr) threadblock_layer<WIDTH, N_ITERS, __half, true>(Activation::None, act_shmem, weights_first_layer, dL_dinput + elem_idx_base * WIDTH);
    }

    template <std::uint32_t WIDTH, typename T, Activation ACTIVATION>
    void mlp_fused_backward(cudaStream_t stream, const legacy::GPUMatrix<T, legacy::RM>& weights_first_layer, const legacy::GPUMatrix<T, legacy::RM>& weights, const legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>& dL_doutput, legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>& temporaries, const legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>& forward, legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>* dL_dinput, const std::uint32_t n_hidden_matmuls) {
        static_assert(std::is_same_v<T, __half>, "The fully fused backward pass only supports __half precision.");
        const std::uint32_t batch_size   = dL_doutput.n();
        const std::uint32_t out_width    = dL_doutput.m();
        constexpr std::uint32_t N_BLOCKS = WIDTH / 16u;
        const std::uint32_t N_ITERS      = WIDTH >= 256u ? 2u : 8u;

        legacy::check_or_throw(forward.n() == batch_size);
        legacy::check_or_throw(batch_size % (16u * N_ITERS) == 0u);
        legacy::check_or_throw(!dL_dinput || dL_dinput->layout() == legacy::RM || dL_dinput->stride() == dL_dinput->m());

        const dim3 threads                    = {32u, N_BLOCKS, 1u};
        const std::uint32_t n_elems_per_block = 16u * N_ITERS;
        const std::uint32_t n_blocks          = (batch_size + n_elems_per_block - 1u) / n_elems_per_block;
        const int shmem_size                  = sizeof(__half) * ((16u * N_ITERS) * (WIDTH + (WIDTH % 16u == 0u ? 8u : 0u)));
        const dim3 blocks                     = {n_blocks, 1u, 1u};

        if (dL_doutput.layout() == legacy::RM) {
            check_shmem_error(cudaFuncSetAttribute(kernel_mlp_fused_backward<WIDTH, N_ITERS, ACTIVATION, nvcuda::wmma::col_major>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
            kernel_mlp_fused_backward<WIDTH, N_ITERS, ACTIVATION, nvcuda::wmma::col_major><<<blocks, threads, shmem_size, stream>>>(dL_doutput.data(), weights.data(), temporaries.data(), forward.data(), dL_dinput ? dL_dinput->data() : nullptr, weights_first_layer.data(), dL_doutput.stride(), batch_size, out_width, n_hidden_matmuls);
        } else {
            check_shmem_error(cudaFuncSetAttribute(kernel_mlp_fused_backward<WIDTH, N_ITERS, ACTIVATION, nvcuda::wmma::row_major>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
            kernel_mlp_fused_backward<WIDTH, N_ITERS, ACTIVATION, nvcuda::wmma::row_major><<<blocks, threads, shmem_size, stream>>>(dL_doutput.data(), weights.data(), temporaries.data(), forward.data(), dL_dinput ? dL_dinput->data() : nullptr, weights_first_layer.data(), dL_doutput.stride(), batch_size, out_width, n_hidden_matmuls);
        }
    }

    template <std::uint32_t WIDTH, std::uint32_t N_ITERS, typename OUT_T, typename INPUT_LAYOUT>
    __device__ void threadblock_input_layer_forward_dynamic(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ input_threadblock, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out_intermediate_threadblock_this_layer, const std::uint32_t in_width, const std::uint32_t batch_size) {
        constexpr std::uint32_t SKEW       = WIDTH % 16u == 0u ? 8u : 0u;
        constexpr std::uint32_t INPUT_SKEW = 8u;
        constexpr std::uint32_t N_BLOCKS   = WIDTH / 16u;

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, INPUT_LAYOUT> act_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> weights_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, OUT_T> result_frag[N_ITERS];

        const std::uint32_t li          = threadIdx.x;
        const std::uint32_t wi          = threadIdx.y;
        const std::uint32_t lane_offset = (8u * li) % WIDTH;
        const std::uint32_t row         = (8u * li + wi * 8u * 32u) / WIDTH;
        const std::uint32_t weights_col = 16u * wi;

        __half* __restrict__ weights_shmem   = act_shmem + 16u * (in_width + INPUT_SKEW);
        const std::uint32_t n_elems_per_load = N_BLOCKS * 32u * 8u;
        const std::uint32_t thread_elem_idx  = (li + wi * 32u) * 8u;
        const std::uint32_t n_elems_b        = WIDTH * in_width;

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t idx = thread_elem_idx; idx < n_elems_b; idx += n_elems_per_load) {
            const std::uint32_t idx_skewed                       = idx + idx / in_width * INPUT_SKEW;
            *reinterpret_cast<int4*>(&weights_shmem[idx_skewed]) = *reinterpret_cast<const int4*>(&weights_this_layer[idx]);
        }

        const std::uint32_t n_tensor_ops = in_width / 16u;
        if constexpr (std::is_same_v<INPUT_LAYOUT, nvcuda::wmma::col_major>) __syncthreads();

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t l = 0u; l < N_ITERS; ++l) {
            if constexpr (std::is_same_v<INPUT_LAYOUT, nvcuda::wmma::row_major>) {
                const std::uint32_t n_elems_a = 16u * in_width;

                TCNN_PRAGMA_UNROLL
                for (std::uint32_t idx = thread_elem_idx; idx < n_elems_a; idx += n_elems_per_load) {
                    const std::uint32_t idx_skewed                   = idx + idx / in_width * INPUT_SKEW;
                    *reinterpret_cast<int4*>(&act_shmem[idx_skewed]) = *reinterpret_cast<const int4*>(&input_threadblock[l * n_elems_a + idx]);
                }

                __syncthreads();
            }

            nvcuda::wmma::fill_fragment(result_frag[l], 0.0f);
            TCNN_PRAGMA_UNROLL
            for (std::uint32_t i = 0u; i < n_tensor_ops; ++i) {
                if constexpr (std::is_same_v<INPUT_LAYOUT, nvcuda::wmma::row_major>)
                    nvcuda::wmma::load_matrix_sync(act_frag, act_shmem + 16u * i, in_width + INPUT_SKEW);
                else
                    nvcuda::wmma::load_matrix_sync(act_frag, input_threadblock + 16u * i * batch_size + 16u * l, batch_size);

                nvcuda::wmma::load_matrix_sync(weights_frag, weights_shmem + 16u * i + weights_col * (in_width + INPUT_SKEW), in_width + INPUT_SKEW);
                nvcuda::wmma::mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);
            }

            if constexpr (std::is_same_v<INPUT_LAYOUT, nvcuda::wmma::row_major>) __syncthreads();
            warp_activation<__half>(activation, result_frag[l], result_frag[l]);
        }

        if constexpr (std::is_same_v<INPUT_LAYOUT, nvcuda::wmma::col_major>) __syncthreads();

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t l = 0u; l < N_ITERS; ++l) nvcuda::wmma::store_matrix_sync(act_shmem + weights_col + (16u * l) * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, nvcuda::wmma::mem_row_major);

        if (out_intermediate_threadblock_this_layer != nullptr) {
            __syncthreads();

            TCNN_PRAGMA_UNROLL
            for (std::uint32_t i = 0u; i < N_ITERS; ++i) *reinterpret_cast<int4*>(&out_intermediate_threadblock_this_layer[lane_offset + (row + 16u * i) * WIDTH]) = *reinterpret_cast<int4*>(&act_shmem[lane_offset + (row + 16u * i) * (WIDTH + SKEW)]);
        }
    }

    template <std::uint32_t WIDTH, std::uint32_t N_ITERS, typename OUT_T>
    __device__ void threadblock_last_layer_forward(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out, const std::uint32_t output_stride, const nvcuda::wmma::layout_t output_layout) {
        constexpr std::uint32_t SKEW     = WIDTH % 16u == 0u ? 8u : 0u;
        constexpr std::uint32_t N_BLOCKS = WIDTH / 16u;

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> act_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> weights_frag[N_BLOCKS];
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, OUT_T> result_frag;

        const std::uint32_t li = threadIdx.x;
        const std::uint32_t wi = threadIdx.y;

        __half* __restrict__ weights_shmem = act_shmem + N_ITERS * 16u * (WIDTH + SKEW);
        const std::uint32_t weights_row    = (8u * li) % WIDTH;
        const std::uint32_t weights_col    = (8u * li + 8u * 32u * wi) / WIDTH;

        *reinterpret_cast<int4*>(&weights_shmem[weights_row + weights_col * (WIDTH + SKEW)]) = *reinterpret_cast<const int4*>(&weights_this_layer[weights_row + weights_col * WIDTH]);
        __syncthreads();

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t i = 0u; i < N_BLOCKS; ++i) nvcuda::wmma::load_matrix_sync(weights_frag[i], weights_shmem + 16u * i, WIDTH + SKEW);

        for (std::uint32_t idx = wi; idx < N_ITERS; idx += N_BLOCKS) {
            nvcuda::wmma::fill_fragment(result_frag, 0.0f);

            TCNN_PRAGMA_UNROLL
            for (std::uint32_t i = 0u; i < N_BLOCKS; ++i) {
                nvcuda::wmma::load_matrix_sync(act_frag, act_shmem + 16u * i + (16u * idx) * (WIDTH + SKEW), WIDTH + SKEW);
                nvcuda::wmma::mma_sync(result_frag, act_frag, weights_frag[i], result_frag);
            }

            warp_activation<__half>(activation, result_frag, result_frag);
            if (output_layout == nvcuda::wmma::mem_row_major)
                nvcuda::wmma::store_matrix_sync(out + idx * 16u * output_stride, result_frag, output_stride, output_layout);
            else
                nvcuda::wmma::store_matrix_sync(out + idx * 16u, result_frag, output_stride, output_layout);
        }
    }

    template <std::uint32_t WIDTH, std::uint32_t N_ITERS>
    __device__ void threadblock_write_output_static(const __half* __restrict__ act_shmem, __half* __restrict__ output_threadblock) {
        constexpr std::uint32_t SKEW = WIDTH % 16u == 0u ? 8u : 0u;

        const std::uint32_t li          = threadIdx.x;
        const std::uint32_t wi          = threadIdx.y;
        const std::uint32_t lane_offset = (8u * li) % WIDTH;
        const std::uint32_t row         = (8u * li + wi * 8u * 32u) / WIDTH;

        __syncthreads();

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t i = 0u; i < N_ITERS; ++i) *reinterpret_cast<int4*>(&output_threadblock[lane_offset + (row + 16u * i) * WIDTH]) = *reinterpret_cast<const int4*>(&act_shmem[lane_offset + (row + 16u * i) * (WIDTH + SKEW)]);
    }

    template <std::uint32_t WIDTH, std::uint32_t N_ITERS, typename OUT_T, Activation ACTIVATION, bool INFERENCE>
    __global__ void kernel_mlp_fused(const Activation output_activation, const __half* __restrict__ input, const __half* __restrict__ weights, OUT_T* __restrict__ out_intermediate, OUT_T* __restrict__ out, const std::uint32_t output_stride, const std::uint32_t batch_size, const std::uint32_t in_width, const std::uint32_t out_width, const std::uint32_t n_hidden_matmuls, const nvcuda::wmma::layout_t input_layout, const nvcuda::wmma::layout_t output_layout) {
        extern __shared__ __half shmem[];
        __half* act_shmem = shmem;

        const std::uint32_t elem_idx = 16u * blockIdx.x * N_ITERS;

        if (input_layout == nvcuda::wmma::mem_col_major || in_width != WIDTH) {
            if (input_layout == nvcuda::wmma::mem_row_major)
                threadblock_input_layer_forward_dynamic<WIDTH, N_ITERS, OUT_T, nvcuda::wmma::row_major>(ACTIVATION, act_shmem, input + elem_idx * in_width, weights, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr, in_width, batch_size);
            else
                threadblock_input_layer_forward_dynamic<WIDTH, N_ITERS, OUT_T, nvcuda::wmma::col_major>(ACTIVATION, act_shmem, input + elem_idx, weights, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr, in_width, batch_size);
        } else {
            threadblock_load_input_static<WIDTH, N_ITERS>(act_shmem, input + elem_idx * WIDTH);
            threadblock_layer<WIDTH, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr);
        }

        const std::uint32_t first_weights_stride = WIDTH * in_width;
        const std::uint32_t weights_stride       = WIDTH * WIDTH;
        const std::uint32_t layer_stride         = WIDTH * batch_size;

        for (std::uint32_t k = 0u; k < n_hidden_matmuls; ++k) threadblock_layer<WIDTH, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights + first_weights_stride + weights_stride * k, !INFERENCE ? (out_intermediate + layer_stride * (k + 1u) + elem_idx * WIDTH) : nullptr);

        if (out_width > 16u) {
            if (INFERENCE) threadblock_write_output_static<WIDTH, N_ITERS>(act_shmem, out_intermediate + elem_idx * WIDTH);
        } else if (out != nullptr) {
            if (output_layout == nvcuda::wmma::mem_row_major)
                threadblock_last_layer_forward<WIDTH, N_ITERS, OUT_T>(output_activation, act_shmem, weights + first_weights_stride + weights_stride * n_hidden_matmuls, out + elem_idx * output_stride, output_stride, output_layout);
            else
                threadblock_last_layer_forward<WIDTH, N_ITERS, OUT_T>(output_activation, act_shmem, weights + first_weights_stride + weights_stride * n_hidden_matmuls, out + elem_idx, output_stride, output_layout);
        }
    }

    template <std::uint32_t WIDTH, typename T, Activation ACTIVATION, bool INFERENCE>
    void mlp_fused_forward(cudaStream_t stream, Activation output_activation, const legacy::GPUMatrix<T, legacy::RM>& weights, const legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>& input, legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>& output_intermediate, legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>* output, const std::uint32_t n_hidden_layers) {
        static_assert(std::is_same_v<T, __half>, "The fully fused forward pass only supports __half precision.");
        const std::uint32_t batch_size = input.n();
        const std::uint32_t in_width   = input.m();

        constexpr std::uint32_t SKEW         = WIDTH % 16u == 0u ? 8u : 0u;
        constexpr std::uint32_t INPUT_SKEW   = 8u;
        constexpr std::uint32_t N_BLOCK_ROWS = WIDTH / 16u;

        static_assert(WIDTH % 16u == 0u, "Width must be a multiply of 16.");

        legacy::check_or_throw(in_width % 16u == 0u);
        legacy::check_or_throw(weights.m() == WIDTH);
        legacy::check_or_throw(weights.n() % 16u == 0u);
        legacy::check_or_throw(output_intermediate.n() == batch_size);
        legacy::check_or_throw(!output || output->n() == batch_size);
        legacy::check_or_throw(input.layout() == legacy::RM || input.stride() == input.m());

        const std::uint32_t N_ITERS = WIDTH >= 256u ? 2u : 8u;
        if (batch_size % (16u * N_ITERS) != 0u) {
            std::ostringstream stream_message;
            stream_message << "Batch size must be a multiple of " << (16u * N_ITERS) << '.';
            throw std::runtime_error{stream_message.str()};
        }

        const dim3 threads                    = {32u, N_BLOCK_ROWS, 1u};
        const std::uint32_t n_elems_per_block = 16u * N_ITERS;
        const std::uint32_t n_blocks          = (batch_size + n_elems_per_block - 1u) / n_elems_per_block;

        std::size_t shmem_size = sizeof(__half) * (16u + 16u * N_ITERS) * (WIDTH + SKEW);
        if (in_width != WIDTH || input.layout() == legacy::RM) shmem_size = std::max(shmem_size, sizeof(__half) * (WIDTH + 16u) * (in_width + INPUT_SKEW));

        const dim3 blocks = {n_blocks, 1u, 1u};

        check_shmem_error(cudaFuncSetAttribute(kernel_mlp_fused<WIDTH, N_ITERS, __half, ACTIVATION, INFERENCE>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shmem_size)));
        kernel_mlp_fused<WIDTH, N_ITERS, __half, ACTIVATION, INFERENCE>
            <<<blocks, threads, shmem_size, stream>>>(output_activation, input.data(), weights.data(), output_intermediate.data(), output ? output->data() : nullptr, output ? output->stride() : 0u, batch_size, in_width, output ? output->m() : 0u, n_hidden_layers, input.layout() == legacy::RM ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major, output && output->layout() == legacy::RM ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major);
    }

    template <typename T, std::uint32_t WIDTH>
    FullyFusedMLP<T, WIDTH>::FullyFusedMLP(const std::uint32_t input_width, const std::uint32_t output_width, const std::uint32_t n_hidden_layers, const Activation activation, const Activation output_activation) : n_hidden_layers{n_hidden_layers}, input_width{input_width}, output_width{output_width}, activation{activation}, output_activation{output_activation} {
        if (this->n_hidden_layers == 0u) throw std::runtime_error{"FullyFusedMLP requires at least 1 hidden layer (3 layers in total)."};

        n_hidden_matmuls    = this->n_hidden_layers - 1u;
        padded_output_width = legacy::next_multiple(this->output_width, 16u);

        weight_matrices.emplace_back(nullptr, network_width, this->input_width);
        gradient_matrices.emplace_back(nullptr, network_width, this->input_width);

        for (std::uint32_t i = 0u; i < n_hidden_matmuls; ++i) {
            weight_matrices.emplace_back(nullptr, network_width, network_width);
            gradient_matrices.emplace_back(nullptr, network_width, network_width);
        }

        weight_matrices.emplace_back(nullptr, padded_output_width, network_width);
        gradient_matrices.emplace_back(nullptr, padded_output_width, network_width);

        n_params = 0u;
        for (const auto& matrix : weight_matrices) n_params += matrix.n_elements();
    }

    template <typename T, std::uint32_t WIDTH>
    void FullyFusedMLP<T, WIDTH>::inference(cudaStream_t stream, const legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>& input, legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>& output) {
        legacy::check_or_throw(input.m() == input_width);
        legacy::check_or_throw(output.m() == padded_output_width);
        legacy::check_or_throw(input.n() % network::detail::batch_size_granularity == 0u);
        legacy::check_or_throw(input.n() == output.n());
        legacy::check_or_throw(params != nullptr);

        const std::uint32_t batch_size            = input.n();
        legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic> inference_tmp = output_width > 16u ? ngp::legacy::GPUMatrix<T, ngp::legacy::MatrixLayout::Dynamic>{network_width, batch_size, stream, legacy::CM} : ngp::legacy::GPUMatrix<T, ngp::legacy::MatrixLayout::Dynamic>{nullptr, network_width, batch_size, legacy::CM};

        switch (activation) {
        case Activation::None: mlp_fused_forward<WIDTH, T, Activation::None, true>(stream, output_activation, weight_matrices.front(), input, inference_tmp, &output, n_hidden_matmuls); break;
        case Activation::Exponential: mlp_fused_forward<WIDTH, T, Activation::Exponential, true>(stream, output_activation, weight_matrices.front(), input, inference_tmp, &output, n_hidden_matmuls); break;
        case Activation::Sigmoid: mlp_fused_forward<WIDTH, T, Activation::Sigmoid, true>(stream, output_activation, weight_matrices.front(), input, inference_tmp, &output, n_hidden_matmuls); break;
        case Activation::ReLU: mlp_fused_forward<WIDTH, T, Activation::ReLU, true>(stream, output_activation, weight_matrices.front(), input, inference_tmp, &output, n_hidden_matmuls); break;
        case Activation::LeakyReLU: mlp_fused_forward<WIDTH, T, Activation::LeakyReLU, true>(stream, output_activation, weight_matrices.front(), input, inference_tmp, &output, n_hidden_matmuls); break;
        case Activation::Squareplus: mlp_fused_forward<WIDTH, T, Activation::Squareplus, true>(stream, output_activation, weight_matrices.front(), input, inference_tmp, &output, n_hidden_matmuls); break;
        case Activation::Softplus: mlp_fused_forward<WIDTH, T, Activation::Softplus, true>(stream, output_activation, weight_matrices.front(), input, inference_tmp, &output, n_hidden_matmuls); break;
        case Activation::Tanh: mlp_fused_forward<WIDTH, T, Activation::Tanh, true>(stream, output_activation, weight_matrices.front(), input, inference_tmp, &output, n_hidden_matmuls); break;
        default: throw std::runtime_error{"Unsupported activation."};
        }

        if (output_width > 16u) fc_multiply<LastLayer>(stream, weight_matrices.back(), inference_tmp, output, output_activation);
    }

    template <typename T, std::uint32_t WIDTH>
    void FullyFusedMLP<T, WIDTH>::forward(cudaStream_t stream, const legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>& input, legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>* output, Scratch& scratch) {
        legacy::check_or_throw(input.m() == input_width);
        legacy::check_or_throw(!output || output->m() == padded_output_width);
        legacy::check_or_throw(input.n() % network::detail::batch_size_granularity == 0u);
        legacy::check_or_throw(!output || input.n() == output->n());
        legacy::check_or_throw(params != nullptr);

        switch (activation) {
        case Activation::None: mlp_fused_forward<WIDTH, T, Activation::None, false>(stream, output_activation, weight_matrices.front(), input, scratch.forward_hidden.at(0), output, n_hidden_matmuls); break;
        case Activation::Exponential: mlp_fused_forward<WIDTH, T, Activation::Exponential, false>(stream, output_activation, weight_matrices.front(), input, scratch.forward_hidden.at(0), output, n_hidden_matmuls); break;
        case Activation::Sigmoid: mlp_fused_forward<WIDTH, T, Activation::Sigmoid, false>(stream, output_activation, weight_matrices.front(), input, scratch.forward_hidden.at(0), output, n_hidden_matmuls); break;
        case Activation::ReLU: mlp_fused_forward<WIDTH, T, Activation::ReLU, false>(stream, output_activation, weight_matrices.front(), input, scratch.forward_hidden.at(0), output, n_hidden_matmuls); break;
        case Activation::LeakyReLU: mlp_fused_forward<WIDTH, T, Activation::LeakyReLU, false>(stream, output_activation, weight_matrices.front(), input, scratch.forward_hidden.at(0), output, n_hidden_matmuls); break;
        case Activation::Squareplus: mlp_fused_forward<WIDTH, T, Activation::Squareplus, false>(stream, output_activation, weight_matrices.front(), input, scratch.forward_hidden.at(0), output, n_hidden_matmuls); break;
        case Activation::Softplus: mlp_fused_forward<WIDTH, T, Activation::Softplus, false>(stream, output_activation, weight_matrices.front(), input, scratch.forward_hidden.at(0), output, n_hidden_matmuls); break;
        case Activation::Tanh: mlp_fused_forward<WIDTH, T, Activation::Tanh, false>(stream, output_activation, weight_matrices.front(), input, scratch.forward_hidden.at(0), output, n_hidden_matmuls); break;
        default: throw std::runtime_error{"Unsupported activation."};
        }

        if (output && output_width > 16u) fc_multiply<LastLayer>(stream, weight_matrices.back(), scratch.forward_hidden.back(), *output, output_activation);
    }

    template <typename T, std::uint32_t WIDTH>
    void FullyFusedMLP<T, WIDTH>::backward(cudaStream_t stream, const cudaStream_t* aux_streams, const cudaEvent_t* aux_events, const std::uint32_t n_aux_streams, Scratch& scratch, const legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>& input, const legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>& output, const legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>& dL_doutput, legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>* dL_dinput, const network::detail::GradientMode param_gradients_mode) {
        legacy::check_or_throw(input.m() == input_width);
        legacy::check_or_throw(output.m() == padded_output_width);
        legacy::check_or_throw(dL_doutput.m() == padded_output_width);
        legacy::check_or_throw(!dL_dinput || dL_dinput->m() == input_width);
        legacy::check_or_throw(input.n() % network::detail::batch_size_granularity == 0u);
        legacy::check_or_throw(input.n() == output.n());
        legacy::check_or_throw(input.n() == dL_doutput.n());
        legacy::check_or_throw(!dL_dinput || input.n() == dL_dinput->n());
        legacy::check_or_throw(params != nullptr);
        if (param_gradients_mode != network::detail::GradientMode::Ignore) legacy::check_or_throw(gradients != nullptr);
        if (param_gradients_mode != network::detail::GradientMode::Ignore) legacy::check_or_throw(aux_streams != nullptr);
        if (param_gradients_mode != network::detail::GradientMode::Ignore) legacy::check_or_throw(aux_events != nullptr);
        if (param_gradients_mode != network::detail::GradientMode::Ignore) legacy::check_or_throw(n_aux_streams >= n_hidden_matmuls + 2u);

        const std::uint32_t batch_size = dL_doutput.n();
        if (output_activation != Activation::None) activation_backward_output_gpu(stream, dL_doutput.n_elements(), output_activation, output.data(), dL_doutput.data(), scratch.backward_output.data());

        const float param_gradient_beta = param_gradients_mode == network::detail::GradientMode::Accumulate ? 1.0f : 0.0f;
        std::uint32_t aux_stream_index = 0u;
        const std::uint32_t split_k_factor                = batch_size / std::min(1u << 12u, batch_size);
        const legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>& tmp_dL_doutput = output_activation == Activation::None ? dL_doutput : scratch.backward_output;

        std::uint32_t tmp_idx          = n_hidden_matmuls;
        std::uint32_t backward_tmp_idx = 0u;

        if (param_gradients_mode != network::detail::GradientMode::Ignore) {
            legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic> output_gradient{gradient_matrices.back().data(), gradient_matrices.back().m(), gradient_matrices.back().n(), gradient_matrices.back().layout(), gradient_matrices.back().stride()};
            legacy::cuda_check(cudaEventRecord(aux_events[aux_stream_index], stream));
            legacy::cuda_check(cudaStreamWaitEvent(aux_streams[aux_stream_index], aux_events[aux_stream_index], 0));
            fc_multiply_split_k<LastLayerK>(aux_streams[aux_stream_index], tmp_dL_doutput, scratch.forward_hidden.at(tmp_idx).transposed(), output_gradient, split_k_factor, param_gradient_beta);
            ++aux_stream_index;
        }

        if (output_width > 16u) fc_multiply<FullLayer>(stream, weight_matrices.back().transposed(), tmp_dL_doutput, scratch.forward_hidden.at(tmp_idx), scratch.backward_hidden.at(backward_tmp_idx), activation, true);

        legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>* dL_dinput_fused = input.m() == scratch.forward_hidden.at(0).m() && input.layout() == legacy::CM ? dL_dinput : nullptr;

        switch (activation) {
        case Activation::None: mlp_fused_backward<WIDTH, T, Activation::None>(stream, weight_matrices.front(), weight_matrices.at(1u), tmp_dL_doutput, scratch.backward_hidden.at(backward_tmp_idx), scratch.forward_hidden.at(0), dL_dinput_fused, n_hidden_matmuls); break;
        case Activation::Exponential: mlp_fused_backward<WIDTH, T, Activation::Exponential>(stream, weight_matrices.front(), weight_matrices.at(1u), tmp_dL_doutput, scratch.backward_hidden.at(backward_tmp_idx), scratch.forward_hidden.at(0), dL_dinput_fused, n_hidden_matmuls); break;
        case Activation::Sigmoid: mlp_fused_backward<WIDTH, T, Activation::Sigmoid>(stream, weight_matrices.front(), weight_matrices.at(1u), tmp_dL_doutput, scratch.backward_hidden.at(backward_tmp_idx), scratch.forward_hidden.at(0), dL_dinput_fused, n_hidden_matmuls); break;
        case Activation::ReLU: mlp_fused_backward<WIDTH, T, Activation::ReLU>(stream, weight_matrices.front(), weight_matrices.at(1u), tmp_dL_doutput, scratch.backward_hidden.at(backward_tmp_idx), scratch.forward_hidden.at(0), dL_dinput_fused, n_hidden_matmuls); break;
        case Activation::LeakyReLU: mlp_fused_backward<WIDTH, T, Activation::LeakyReLU>(stream, weight_matrices.front(), weight_matrices.at(1u), tmp_dL_doutput, scratch.backward_hidden.at(backward_tmp_idx), scratch.forward_hidden.at(0), dL_dinput_fused, n_hidden_matmuls); break;
        case Activation::Squareplus: mlp_fused_backward<WIDTH, T, Activation::Squareplus>(stream, weight_matrices.front(), weight_matrices.at(1u), tmp_dL_doutput, scratch.backward_hidden.at(backward_tmp_idx), scratch.forward_hidden.at(0), dL_dinput_fused, n_hidden_matmuls); break;
        case Activation::Softplus: mlp_fused_backward<WIDTH, T, Activation::Softplus>(stream, weight_matrices.front(), weight_matrices.at(1u), tmp_dL_doutput, scratch.backward_hidden.at(backward_tmp_idx), scratch.forward_hidden.at(0), dL_dinput_fused, n_hidden_matmuls); break;
        case Activation::Tanh: mlp_fused_backward<WIDTH, T, Activation::Tanh>(stream, weight_matrices.front(), weight_matrices.at(1u), tmp_dL_doutput, scratch.backward_hidden.at(backward_tmp_idx), scratch.forward_hidden.at(0), dL_dinput_fused, n_hidden_matmuls); break;
        default: throw std::runtime_error{"Unsupported activation."};
        }

        tmp_idx -= 1u;
        ++backward_tmp_idx;

        for (std::uint32_t i = 0u; i < n_hidden_matmuls; ++i) {
            const std::uint32_t matrix_idx = n_hidden_matmuls - i - 1u;

            if (param_gradients_mode != network::detail::GradientMode::Ignore) {
                legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic> gradient_matrix{gradient_matrices.at(1u + matrix_idx).data(), gradient_matrices.at(1u + matrix_idx).m(), gradient_matrices.at(1u + matrix_idx).n(), gradient_matrices.at(1u + matrix_idx).layout(), gradient_matrices.at(1u + matrix_idx).stride()};
                legacy::cuda_check(cudaEventRecord(aux_events[aux_stream_index], stream));
                legacy::cuda_check(cudaStreamWaitEvent(aux_streams[aux_stream_index], aux_events[aux_stream_index], 0));
                fc_multiply_split_k<FullLayerK>(aux_streams[aux_stream_index], scratch.backward_hidden.at(backward_tmp_idx - 1u), scratch.forward_hidden.at(tmp_idx).transposed(), gradient_matrix, split_k_factor, param_gradient_beta);
                ++aux_stream_index;
            }

            tmp_idx -= 1u;
            ++backward_tmp_idx;
        }

        if (param_gradients_mode != network::detail::GradientMode::Ignore) {
            legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic> input_gradient{gradient_matrices.front().data(), gradient_matrices.front().m(), gradient_matrices.front().n(), gradient_matrices.front().layout(), gradient_matrices.front().stride()};
            legacy::cuda_check(cudaEventRecord(aux_events[aux_stream_index], stream));
            legacy::cuda_check(cudaStreamWaitEvent(aux_streams[aux_stream_index], aux_events[aux_stream_index], 0));
            fc_multiply_split_k<FullLayerK>(aux_streams[aux_stream_index], scratch.backward_hidden.at(backward_tmp_idx - 1u), input.transposed(), input_gradient, split_k_factor, param_gradient_beta);
            ++aux_stream_index;
        }

        if (dL_dinput && !dL_dinput_fused) fc_multiply<FullLayer>(stream, weight_matrices.front().transposed(), scratch.backward_hidden.at(backward_tmp_idx - 1u), *dL_dinput);
        for (std::uint32_t i = 0u; i < aux_stream_index; ++i) {
            legacy::cuda_check(cudaEventRecord(aux_events[i], aux_streams[i]));
            legacy::cuda_check(cudaStreamWaitEvent(stream, aux_events[i], 0));
        }
    }

    template <typename T, std::uint32_t WIDTH>
    void FullyFusedMLP<T, WIDTH>::prepare_scratch(cudaStream_t stream, const std::uint32_t batch_size, const legacy::MatrixLayout output_layout, Scratch& scratch) {
        scratch.forward_hidden.resize(n_hidden_layers);
        scratch.backward_hidden.resize(n_hidden_layers);

        std::size_t shared_forward_bytes  = 0u;
        std::size_t shared_backward_bytes = 0u;
        for (std::uint32_t i = 0u; i < n_hidden_layers; ++i) {
            scratch.forward_hidden[i].set_size_unsafe(network_width, batch_size);
            scratch.backward_hidden[i].set_size_unsafe(network_width, batch_size);
            shared_forward_bytes += scratch.forward_hidden[i].n_bytes();
            shared_backward_bytes += scratch.backward_hidden[i].n_bytes();
        }

        scratch.forward_alloc  = legacy::GpuAllocation{shared_forward_bytes, stream};
        scratch.backward_alloc = legacy::GpuAllocation{shared_backward_bytes, stream};

        void* forward_base          = scratch.forward_alloc.data();
        void* backward_base         = scratch.backward_alloc.data();
        std::size_t forward_offset  = 0u;
        std::size_t backward_offset = 0u;
        for (std::uint32_t i = 0u; i < n_hidden_layers; ++i) {
            scratch.forward_hidden[i].set_data_unsafe(static_cast<char*>(forward_base) + forward_offset);
            scratch.backward_hidden[i].set_data_unsafe(static_cast<char*>(backward_base) + backward_offset);
            forward_offset += scratch.forward_hidden[i].n_bytes();
            backward_offset += scratch.backward_hidden[i].n_bytes();
        }

        if (output_activation != Activation::None)
            scratch.backward_output = {padded_output_width, batch_size, stream, output_layout};
        else
            scratch.backward_output = {};
    }

    template <typename T, std::uint32_t WIDTH>
    void FullyFusedMLP<T, WIDTH>::initialize_params(legacy::math::pcg32& rnd, float* params_full_precision, const float scale) {
        std::vector<legacy::GPUMatrix<float, legacy::RM>> weight_matrices_full_precision;
        weight_matrices_full_precision.emplace_back(params_full_precision, network_width, input_width);
        params_full_precision += weight_matrices_full_precision.back().n_elements();

        for (std::uint32_t i = 0u; i < n_hidden_matmuls; ++i) {
            weight_matrices_full_precision.emplace_back(params_full_precision, network_width, network_width);
            params_full_precision += weight_matrices_full_precision.back().n_elements();
        }

        weight_matrices_full_precision.emplace_back(params_full_precision, padded_output_width, network_width);
        for (auto& matrix : weight_matrices_full_precision) matrix.initialize_xavier_uniform(rnd, scale);
    }

    template class FullyFusedMLP<__half, 16u>;
    template class FullyFusedMLP<__half, 32u>;
    template class FullyFusedMLP<__half, 64u>;
    template class FullyFusedMLP<__half, 128u>;

} // namespace ngp::mlp
