/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <miopen/gemm_v2.hpp>
#include <miopen/logger.hpp>
#include <miopen/env.hpp>

#if MIOPEN_USE_ROCBLAS
//#include <half.hpp>
#include <rocblas.h>
#include <miopen/hipoc_kernel.hpp>
#include <miopen/perf_field.hpp>
#endif

#if MIOPEN_USE_ROCBLAS
#define ROCBLAS_TIMING_MEMSET_SIZE (10 * 1024 * 1024)
#endif

MIOPEN_DECLARE_ENV_VAR(MIOPEN_GEMM_ENFORCE_BACKEND)

namespace miopen {

#if MIOPEN_USE_ROCBLAS
// Enqueue gpu memset for rocblas kernel timing purpose
// Be careful, will set mem to 0
static void
dummy_memset(Handle& handle, Data_t mem, std::size_t mem_len, miopenDataType_t data_type)
{
    MIOPEN_LOG_I2("dummy gpu memset");

    std::size_t data_size = 0;

    switch(data_type)
    {
       case miopenHalf:
           break;
       case miopenFloat:
       {
           data_size = sizeof(float);
           break;
       }
    }

    std::size_t sz = mem_len * data_size;

    for(std::size_t i = 0; i < ROCBLAS_TIMING_MEMSET_SIZE; i += sz)
#ifdef DGPU
       hipMemsetAsync(mem, 0, sz, handle.GetStream());
#else
       memset(mem, 0, sz);
#endif
}
#endif

// hacks: control GEMM backend by enviroment variable and build option
// very nasty
static GemmBackend_t enforce_gemm_backend(miopenDataType_t data_type,
                                          GemmBackend_t gemm_backend_preferred)
{
    GemmBackend_t gemm_backend_enforced = GemmBackend_t::nogemmbackend;
    GemmBackend_t gemm_backend_env      = GemmBackend_t::nogemmbackend;

    // enforce backend based on env variable
    switch(Value(MIOPEN_GEMM_ENFORCE_BACKEND{}))
    {
    case 1: gemm_backend_env  = GemmBackend_t::rocblas; break;
    case 2: gemm_backend_env  = GemmBackend_t::miopengemm; break;
    case 3: gemm_backend_env  = GemmBackend_t::nogemmbackend; break;
    default: gemm_backend_env = gemm_backend_preferred;
    }

// make sure backend chosen based on env variable is suppported
#if MIOPEN_USE_ROCBLAS and MIOPEN_USE_MIOPENGEMM
    switch(gemm_backend_env)
    {
    case GemmBackend_t::nogemmbackend: gemm_backend_enforced = GemmBackend_t::nogemmbackend; break;
    case GemmBackend_t::rocblas: gemm_backend_enforced       = GemmBackend_t::rocblas; break;
    case GemmBackend_t::miopengemm:
        gemm_backend_enforced =
            (data_type == miopenFloat) ? GemmBackend_t::miopengemm : GemmBackend_t::rocblas;
        break;
    }
#elif MIOPEN_USE_ROCBLAS
    switch(gemm_backend_env)
    {
    case GemmBackend_t::nogemmbackend: gemm_backend_enforced = GemmBackend_t::nogemmbackend; break;
    case GemmBackend_t::rocblas:
    case GemmBackend_t::miopengemm: gemm_backend_enforced = GemmBackend_t::rocblas; break;
    }
#elif MIOPEN_USE_MIOPENGEMM
    switch(gemm_backend_env)
    {
    case GemmBackend_t::nogemmbackend: gemm_backend_enforced = GemmBackend_t::nogemmbackend; break;
    case GemmBackend_t::rocblas:
    case GemmBackend_t::miopengemm:
        gemm_backend_enforced =
            (data_type == miopenFloat) ? GemmBackend_t::miopengemm : GemmBackend_t::nogemmbackend;
        break;
    }
#else
    gemm_backend_enforced = GemmBackend_t::nogemmbackend;
#endif

    return gemm_backend_enforced;
}

miopenStatus_t CallGemm(Handle& handle,
                        GemmDescriptor gemm_desc,
                        ConstData_t A,
                        int a_offset,
                        ConstData_t B,
                        int b_offset,
                        Data_t C,
                        int c_offset,
                        std::string* kcache_key,
                        bool enqueue_dummy_kernel,
                        GemmBackend_t gemm_backend)
{
#if !MIOPEN_USE_ROCBLAS
    (void)enqueue_dummy_kernel;
#endif

    gemm_backend = enforce_gemm_backend(gemm_desc.dataType, gemm_backend);
    // do row-to-column major conversion here
    if(!gemm_desc.isColMajor)
    {
        gemm_desc.isColMajor = true;
        std::swap(A, B);
        std::swap(a_offset, b_offset);
        std::swap(gemm_desc.transA, gemm_desc.transB);
        std::swap(gemm_desc.m, gemm_desc.n);
        std::swap(gemm_desc.lda, gemm_desc.ldb);
    }

    switch(gemm_backend)
    {
       case GemmBackend_t::nogemmbackend: return miopenStatusNotImplemented;
       case GemmBackend_t::miopengemm: return miopenStatusNotImplemented;
#if MIOPEN_USE_ROCBLAS
       case GemmBackend_t::rocblas: 
       {
           MIOPEN_LOG_FUNCTION("rocBLAS");

           HipEventPtr start = nullptr;
           HipEventPtr stop  = nullptr;
           if(handle.IsProfilingEnabled())
           {
               if(enqueue_dummy_kernel)
               {
                   dummy_memset(handle, C, gemm_desc.m * gemm_desc.n, gemm_desc.dataType);
               }

               start = make_hip_event();
               stop  = make_hip_event();
               hipEventRecord(start.get(), handle.GetStream());
           }

           rocblas_status rb_status = rocblas_status::rocblas_status_internal_error;

           switch(gemm_desc.dataType)
           {
               case miopenHalf: return miopenStatusNotImplemented;
               case miopenFloat:
               {
                   float alpha = gemm_desc.alpha;
                   float beta  = gemm_desc.beta;

                   rocblas_set_stream(handle.rhandle.get(), handle.GetStream());

                   rb_status   = rocblas_sgemm(
                       handle.rhandle.get(),
                       gemm_desc.transA ? rocblas_operation_transpose : rocblas_operation_none,
                       gemm_desc.transB ? rocblas_operation_transpose : rocblas_operation_none,
                       gemm_desc.m,
                       gemm_desc.n,
                       gemm_desc.k,
                       &alpha,
                       static_cast<const float*>(A) + a_offset,
                       gemm_desc.lda,
                       static_cast<const float*>(B) + b_offset,
                       gemm_desc.ldb,
                       &beta,
                       static_cast<float*>(C) + c_offset,
                       gemm_desc.ldc);
               }
               break;
           }

           if(handle.IsProfilingEnabled())
           {
               hipEventRecord(stop.get(), handle.GetStream());
               hipEventSynchronize(stop.get());
               float mS = 0;
               hipEventElapsedTime(&mS, start.get(), stop.get());
               handle.ResetKernelTime();
               handle.AccumKernelTime(mS);
           }

           if(kcache_key != nullptr)
               *kcache_key = FindDbData::GetUnusedKCacheKey();

           if(rb_status != rocblas_status::rocblas_status_success)
               MIOPEN_THROW(miopenStatusInternalError, "rocBlas error encountered");

           return miopenStatusSuccess;
       }
#endif
    }

    return miopenStatusUnknownError;
}

// y = w * Im2Col(x)
GemmDescriptor CreateGemmDescriptorConvFwd(const TensorDescriptor& wDesc,
                                           const TensorDescriptor& xDesc,
                                           const TensorDescriptor& yDesc)
{
#ifndef NDEBUG
    assert(wDesc.GetType() == xDesc.GetType() && wDesc.GetType() == yDesc.GetType());
#endif

    int in_c;
    std::tie(std::ignore, in_c, std::ignore, std::ignore) = tien<4>(xDesc.GetLengths());

    int wei_n, wei_h, wei_w;
    std::tie(wei_n, std::ignore, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(yDesc.GetLengths());

    bool isColMajor       = false;
    bool transA           = false;
    bool transB           = false;
    int m                 = wei_n;
    int n                 = out_h * out_w;
    int k                 = in_c * wei_h * wei_w;
    int lda               = k;
    int ldb               = n;
    int ldc               = n;
    int batch_count       = 1;
    long long int strideA = 0;
    long long int strideB = 0;
    long long int strideC = 0;
    float alpha           = 1.;
    float beta            = 0.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          xDesc.GetType()};
}

// dx = Col2Im(transpose(w) * dy)
GemmDescriptor CreateGemmDescriptorConvBwdData(const TensorDescriptor& wDesc,
                                               const TensorDescriptor& dyDesc,
                                               const TensorDescriptor& dxDesc)
{
#ifndef NDEBUG
    assert(wDesc.GetType() == dxDesc.GetType() && wDesc.GetType() == dyDesc.GetType());
#endif

    int in_c;
    std::tie(std::ignore, in_c, std::ignore, std::ignore) = tien<4>(dxDesc.GetLengths());

    int wei_n, wei_h, wei_w;
    std::tie(wei_n, std::ignore, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(dyDesc.GetLengths());

    bool isColMajor       = false;
    bool transA           = true;
    bool transB           = false;
    int m                 = in_c * wei_h * wei_w;
    int n                 = out_h * out_w;
    int k                 = wei_n;
    int lda               = m;
    int ldb               = n;
    int ldc               = n;
    int batch_count       = 1;
    long long int strideA = 0;
    long long int strideB = 0;
    long long int strideC = 0;
    float alpha           = 1.;
    float beta            = 0.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          dxDesc.GetType()};
}

// dw = dy * transpose(Im2Col(x))
GemmDescriptor CreateGemmDescriptorConvBwdWeight(const TensorDescriptor& dyDesc,
                                                 const TensorDescriptor& xDesc,
                                                 const TensorDescriptor& dwDesc)
{
#ifndef NDEBUG
    assert(dwDesc.GetType() == xDesc.GetType() && dwDesc.GetType() == dyDesc.GetType());
#endif

    int in_c;
    std::tie(std::ignore, in_c, std::ignore, std::ignore) = tien<4>(xDesc.GetLengths());

    int wei_n, wei_h, wei_w;
    std::tie(wei_n, std::ignore, wei_h, wei_w) = tien<4>(dwDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(dyDesc.GetLengths());

    bool isColMajor       = false;
    bool transA           = false;
    bool transB           = true;
    int m                 = wei_n;
    int n                 = in_c * wei_h * wei_w;
    int k                 = out_h * out_w;
    int lda               = k;
    int ldb               = k;
    int ldc               = n;
    int batch_count       = 1;
    long long int strideA = 0;
    long long int strideB = 0;
    long long int strideC = 0;
    float alpha           = 1.;
    float beta            = 1.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          xDesc.GetType()};
}

// y = CNHW2NCHW(w * NCHW2CNHW(x))
GemmDescriptor CreateGemmDescriptorConvCNHWFwd(const TensorDescriptor& wDesc,
                                               const TensorDescriptor& xDesc,
                                               const TensorDescriptor& yDesc)
{
#ifndef NDEBUG
    assert(wDesc.GetType() == xDesc.GetType() && wDesc.GetType() == yDesc.GetType());
#endif

    int in_n, in_c;
    std::tie(in_n, in_c, std::ignore, std::ignore) = tien<4>(xDesc.GetLengths());

    int wei_n;
    std::tie(wei_n, std::ignore, std::ignore, std::ignore) = tien<4>(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(yDesc.GetLengths());

    bool isColMajor       = false;
    bool transA           = false;
    bool transB           = false;
    int m                 = wei_n;
    int n                 = in_n * out_h * out_w;
    int k                 = in_c;
    int lda               = k;
    int ldb               = n;
    int ldc               = n;
    int batch_count       = 1;
    long long int strideA = 0;
    long long int strideB = 0;
    long long int strideC = 0;
    float alpha           = 1.;
    float beta            = 0.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          xDesc.GetType()};
}

// dx = CNHW2NCHW(transpose(w) * NCHW2CNHW(dy))
GemmDescriptor CreateGemmDescriptorConvCNHWBwdData(const TensorDescriptor& wDesc,
                                                   const TensorDescriptor& dyDesc,
                                                   const TensorDescriptor& dxDesc)
{
#ifndef NDEBUG
    assert(wDesc.GetType() == dxDesc.GetType() && wDesc.GetType() == dyDesc.GetType());
#endif

    int in_n, in_c;
    std::tie(in_n, in_c, std::ignore, std::ignore) = tien<4>(dxDesc.GetLengths());

    int wei_n;
    std::tie(wei_n, std::ignore, std::ignore, std::ignore) = tien<4>(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(dyDesc.GetLengths());

    bool isColMajor       = false;
    bool transA           = true;
    bool transB           = false;
    int m                 = in_c;
    int n                 = in_n * out_h * out_w;
    int k                 = wei_n;
    int lda               = m;
    int ldb               = n;
    int ldc               = n;
    int batch_count       = 1;
    long long int strideA = 0;
    long long int strideB = 0;
    long long int strideC = 0;
    float alpha           = 1.;
    float beta            = 0.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          dxDesc.GetType()};
}

// y[i] = w * x[i], i is batch id
GemmDescriptor CreateGemmStridedBatchedDescriptorConv1x1Fwd(const TensorDescriptor& wDesc,
                                                            const TensorDescriptor& xDesc,
                                                            const TensorDescriptor& yDesc)
{
#ifndef NDEBUG
    assert(wDesc.GetType() == xDesc.GetType() && wDesc.GetType() == yDesc.GetType());
#else
    (void)yDesc;
#endif

    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tien<4>(xDesc.GetLengths());

    int wei_n;
    std::tie(wei_n, std::ignore, std::ignore, std::ignore) = tien<4>(wDesc.GetLengths());

    bool isColMajor       = false;
    bool transA           = false;
    bool transB           = false;
    int m                 = wei_n;
    int n                 = in_h * in_w;
    int k                 = in_c;
    int lda               = k;
    int ldb               = n;
    int ldc               = n;
    int batch_count       = in_n;
    long long int strideA = 0;
    long long int strideB = k * n;
    long long int strideC = m * n;
    float alpha           = 1.;
    float beta            = 0.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          xDesc.GetType()};
}

// dx[i] = transpose(w) * dy[i], i is batch id
GemmDescriptor CreateGemmStridedBatchedDescriptorConv1x1BwdData(const TensorDescriptor& wDesc,
                                                                const TensorDescriptor& dyDesc,
                                                                const TensorDescriptor& dxDesc)
{
#ifndef NDEBUG
    assert(wDesc.GetType() == dxDesc.GetType() && wDesc.GetType() == dyDesc.GetType());
#else
    (void)dyDesc;
#endif

    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tien<4>(dxDesc.GetLengths());

    int wei_n;
    std::tie(wei_n, std::ignore, std::ignore, std::ignore) = tien<4>(wDesc.GetLengths());

    bool isColMajor       = false;
    bool transA           = true;
    bool transB           = false;
    int m                 = in_c;
    int n                 = in_h * in_w;
    int k                 = wei_n;
    int lda               = m;
    int ldb               = n;
    int ldc               = n;
    int batch_count       = in_n;
    long long int strideA = 0;
    long long int strideB = k * n;
    long long int strideC = m * n;
    float alpha           = 1.;
    float beta            = 0;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          dxDesc.GetType()};
}

// dw = sum_over_batch(dy[i] * transpose(x[i])), i is batch id
GemmDescriptor CreateGemmStridedBatchedDescriptorConv1x1BwdWeight(const TensorDescriptor& dyDesc,
                                                                  const TensorDescriptor& xDesc,
                                                                  const TensorDescriptor& dwDesc)
{
#ifndef NDEBUG
    assert(dwDesc.GetType() == xDesc.GetType() && dwDesc.GetType() == dyDesc.GetType());
#else
    (void)dyDesc;
#endif

    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tien<4>(xDesc.GetLengths());

    int wei_n;
    std::tie(wei_n, std::ignore, std::ignore, std::ignore) = tien<4>(dwDesc.GetLengths());

    bool isColMajor       = false;
    bool transA           = false;
    bool transB           = true;
    int m                 = wei_n;
    int n                 = in_c;
    int k                 = in_h * in_w;
    int lda               = k;
    int ldb               = k;
    int ldc               = n;
    int batch_count       = in_n;
    long long int strideA = m * k;
    long long int strideB = k * n;
    long long int strideC = 0;
    float alpha           = 1.;
    float beta            = 1.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          xDesc.GetType()};
}

// y = w * Im2Col(x)
GemmDescriptor CreateGemmDescriptorGroupConvFwd(const TensorDescriptor& wDesc,
                                                const TensorDescriptor& xDesc,
                                                const TensorDescriptor& yDesc,
                                                int groupCount)
{
#ifndef NDEBUG
    assert(wDesc.GetType() == xDesc.GetType() && wDesc.GetType() == yDesc.GetType());
#endif

    int in_c;
    std::tie(std::ignore, in_c, std::ignore, std::ignore) = tien<4>(xDesc.GetLengths());

    int wei_n, wei_h, wei_w;
    std::tie(wei_n, std::ignore, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(yDesc.GetLengths());

    bool isColMajor       = false;
    bool transA           = false;
    bool transB           = false;
    int m                 = wei_n / groupCount;
    int n                 = out_h * out_w;
    int k                 = (in_c / groupCount) * wei_h * wei_w;
    int lda               = k;
    int ldb               = n;
    int ldc               = n;
    int batch_count       = groupCount;
    long long int strideA = m * k;
    long long int strideB = k * n;
    long long int strideC = m * n;
    float alpha           = 1.;
    float beta            = 0.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          xDesc.GetType()};
}

// dx = Col2Im(transpose(w) * dy)
GemmDescriptor CreateGemmDescriptorGroupConvBwdData(const TensorDescriptor& wDesc,
                                                    const TensorDescriptor& dyDesc,
                                                    const TensorDescriptor& dxDesc,
                                                    int groupCount)
{
#ifndef NDEBUG
    assert(wDesc.GetType() == dxDesc.GetType() && wDesc.GetType() == dyDesc.GetType());
#endif

    int in_c;
    std::tie(std::ignore, in_c, std::ignore, std::ignore) = tien<4>(dxDesc.GetLengths());

    int wei_n, wei_h, wei_w;
    std::tie(wei_n, std::ignore, wei_h, wei_w) = tien<4>(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(dyDesc.GetLengths());

    bool isColMajor       = false;
    bool transA           = true;
    bool transB           = false;
    int m                 = (in_c / groupCount) * wei_h * wei_w;
    int n                 = out_h * out_w;
    int k                 = wei_n / groupCount;
    int lda               = m;
    int ldb               = n;
    int ldc               = n;
    int batch_count       = groupCount;
    long long int strideA = m * k;
    long long int strideB = k * n;
    long long int strideC = m * n;
    float alpha           = 1.;
    float beta            = 0.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          dxDesc.GetType()};
}

// dw = dy * transpose(Im2Col(x))
GemmDescriptor CreateGemmDescriptorGroupConvBwdWeight(const TensorDescriptor& dyDesc,
                                                      const TensorDescriptor& xDesc,
                                                      const TensorDescriptor& dwDesc,
                                                      int groupCount)
{
#ifndef NDEBUG
    assert(dwDesc.GetType() == xDesc.GetType() && dwDesc.GetType() == dyDesc.GetType());
#endif

    int in_c;
    std::tie(std::ignore, in_c, std::ignore, std::ignore) = tien<4>(xDesc.GetLengths());

    int wei_n, wei_h, wei_w;
    std::tie(wei_n, std::ignore, wei_h, wei_w) = tien<4>(dwDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(dyDesc.GetLengths());

    bool isColMajor       = false;
    bool transA           = false;
    bool transB           = true;
    int m                 = wei_n / groupCount;
    int n                 = (in_c / groupCount) * wei_h * wei_w;
    int k                 = out_h * out_w;
    int lda               = k;
    int ldb               = k;
    int ldc               = n;
    int batch_count       = groupCount;
    long long int strideA = m * k;
    long long int strideB = k * n;
    long long int strideC = m * n;
    float alpha           = 1.;
    float beta            = 1.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          xDesc.GetType()};
}

// y = CNHW2NCHW(w * NCHW2CNHW(x))
GemmDescriptor CreateGemmDescriptorGroupConvCNHWFwd(const TensorDescriptor& wDesc,
                                                    const TensorDescriptor& xDesc,
                                                    const TensorDescriptor& yDesc,
                                                    int groupCount)
{
#ifndef NDEBUG
    assert(wDesc.GetType() == xDesc.GetType() && wDesc.GetType() == yDesc.GetType());
#endif

    int in_n, in_c;
    std::tie(in_n, in_c, std::ignore, std::ignore) = tien<4>(xDesc.GetLengths());

    int wei_n;
    std::tie(wei_n, std::ignore, std::ignore, std::ignore) = tien<4>(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(yDesc.GetLengths());

    bool isColMajor       = false;
    bool transA           = false;
    bool transB           = false;
    int m                 = wei_n / groupCount;
    int n                 = in_n * out_h * out_w;
    int k                 = in_c / groupCount;
    int lda               = k;
    int ldb               = n;
    int ldc               = n;
    int batch_count       = groupCount;
    long long int strideA = m * k;
    long long int strideB = k * n;
    long long int strideC = m * n;
    float alpha           = 1.;
    float beta            = 0.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          xDesc.GetType()};
}

// dx = CNHW2NCHW(transpose(w) * NCHW2CNHW(dy))
GemmDescriptor CreateGemmDescriptorGroupConvCNHWBwdData(const TensorDescriptor& wDesc,
                                                        const TensorDescriptor& dyDesc,
                                                        const TensorDescriptor& dxDesc,
                                                        int groupCount)
{
#ifndef NDEBUG
    assert(wDesc.GetType() == dxDesc.GetType() && wDesc.GetType() == dyDesc.GetType());
#endif

    int in_n, in_c;
    std::tie(in_n, in_c, std::ignore, std::ignore) = tien<4>(dxDesc.GetLengths());

    int wei_n;
    std::tie(wei_n, std::ignore, std::ignore, std::ignore) = tien<4>(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tien<4>(dyDesc.GetLengths());

    bool isColMajor       = false;
    bool transA           = true;
    bool transB           = false;
    int m                 = in_c / groupCount;
    int n                 = in_n * out_h * out_w;
    int k                 = wei_n / groupCount;
    int lda               = m;
    int ldb               = n;
    int ldc               = n;
    int batch_count       = groupCount;
    long long int strideA = m * k;
    long long int strideB = k * n;
    long long int strideC = m * n;
    float alpha           = 1.;
    float beta            = 0.;

    return GemmDescriptor{isColMajor,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          lda,
                          ldb,
                          ldc,
                          batch_count,
                          strideA,
                          strideB,
                          strideC,
                          alpha,
                          beta,
                          dxDesc.GetType()};
}

} // namespace miopen
