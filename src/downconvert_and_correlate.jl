#=
# This is currently slower than splitting the loop.
# See https://github.com/JuliaSIMD/LoopVectorization.jl/issues/284
function downconvert_and_correlate(
    signal::StructArray{Complex{T}},
    correlator::C,
    code,
    correlator_sample_shifts,
    carrier_frequency,
    sampling_frequency,
    start_phase,
    start_sample,
    num_samples
) where {T, C <: AbstractCorrelator}
    s_re = signal.re; s_im = signal.im
    accumulators = zero_accumulators(get_accumulators(correlator), signal)
    a_re = real.(accumulators)
    a_im = imag.(accumulators)
    @avx for i = start_sample:start_sample + num_samples - 1
        c_im, c_re = sincos(T(2π) * ((i - start_sample) * T(upreferred(carrier_frequency / Hz)) / T(upreferred(sampling_frequency / Hz)) + T(start_phase)))
        d_re = s_re[i] * c_re + s_im[i] * c_im
        d_im = s_im[i] * c_re - s_re[i] * c_im
        for j = 1:length(a_re)
            sample_shift = correlator_sample_shifts[j] - correlator_sample_shifts[1]
            a_re[j] += d_re * code[i + sample_shift]
            a_im[j] += d_im * code[i + sample_shift]
        end
    end
    accumulators_result = complex.(a_re, a_im)
    C(map(+, get_accumulators(correlator), accumulators_result))
end
=#

function gen_carrier_replica_kernel!(
    carrier_replica_re,
    carrier_replica_im,
    carrier_frequency,
    sampling_frequency,
    carrier_phase,
    num_samples
)
    sample_idx   = 1 + ((blockIdx().x - 1) * blockDim().x + (threadIdx().x - 1))
    if sample_idx <= num_samples
        @inbounds carrier_replica_im[sample_idx], carrier_replica_re[sample_idx] = CUDA.sincos(2π * ((sample_idx - 1) * carrier_frequency / sampling_frequency + carrier_phase))
    end
    return nothing
end

function gen_code_replica_kernel!(
    code_replica,
    codes,
    code_frequency,
    sampling_frequency,
    start_code_phase,
    prn,
    num_samples,
    num_of_shifts,
    code_length
)   
    # thread_idx goes from 1:2502
    # sample_idx converted to -1:2500 -> [-1 0 +1]
    thread_idx = 1 + ((blockIdx().x - 1) * blockDim().x + (threadIdx().x - 1))
    if thread_idx <= num_samples 
        @inbounds code_replica[thread_idx] = codes[1+mod(floor(Int32, code_frequency/sampling_frequency * (thread_idx - num_of_shifts) + start_code_phase), code_length), prn]
    end
    
    return nothing   
end

# CUDA Kernel 
function downconvert_and_correlate_kernel(
    res_re,
    res_im,
    signal_re,
    signal_im,
    carrier_re,
    carrier_im,
    code,
    downconverted_signal_re,
    downconverted_signal_im,
    correlator_sample_shifts::SVector,
    num_samples,
    num_ants,#::NumAnts{N}
    num_corrs
)#  where N
    cache = @cuDynamicSharedMem(Float32, (2 * blockDim().x, num_ants, num_corrs))   
    sample_idx   = 1 + ((blockIdx().x - 1) * blockDim().x + (threadIdx().x - 1))
    antenna_idx  = 1 + ((blockIdx().y - 1) * blockDim().y + (threadIdx().y - 1))
    iq_offset = blockDim().x
    cache_index = threadIdx().x - 1 

    accum_re = accum_im = 0.0f0

    if sample_idx <= num_samples && antenna_idx <= num_ants
        # downconvert with the conjugate of the carrier
        #= for i = 1:N

        end =#
        downconverted_signal_re[sample_idx, antenna_idx] = signal_re[sample_idx, antenna_idx] * carrier_re[sample_idx] + signal_im[sample_idx, antenna_idx] * carrier_im[sample_idx]
        downconverted_signal_im[sample_idx, antenna_idx] = signal_im[sample_idx, antenna_idx] * carrier_re[sample_idx] - signal_re[sample_idx, antenna_idx] * carrier_im[sample_idx]
        
        # multiply elementwise with the code
        for (corr_idx, sample_shift) ∈ enumerate(correlator_sample_shifts)
            accum_re += code[sample_idx - 1 + 2 + sample_shift] * downconverted_signal_re[sample_idx, antenna_idx]
            accum_im += code[sample_idx - 1 + 2 + sample_shift] * downconverted_signal_im[sample_idx, antenna_idx]

            cache[1 + cache_index + 0 * iq_offset, antenna_idx, corr_idx] = accum_re
            cache[1 + cache_index + 1 * iq_offset, antenna_idx, corr_idx] = accum_im
        end
    end

    ## Reduction
    # wait until all the accumulators have done writing the results to the cache
    sync_threads()

    i::Int = blockDim().x ÷ 2
    @inbounds while i != 0
        if cache_index < i
            for (corr_idx, _) ∈ enumerate(correlator_sample_shifts)
                cache[1 + cache_index + 0 * iq_offset, antenna_idx, corr_idx] += cache[1 + cache_index + 0 * iq_offset + i, antenna_idx, corr_idx]
                cache[1 + cache_index + 1 * iq_offset, antenna_idx, corr_idx] += cache[1 + cache_index + 1 * iq_offset + i, antenna_idx, corr_idx]
            end
        end
        sync_threads()
        i ÷= 2
    end
    
    if (threadIdx().x - 1) == 0
        for (corr_idx, _) in enumerate(correlator_sample_shifts) 
            res_re[blockIdx().x, antenna_idx, corr_idx] += cache[1 + 0 * iq_offset, antenna_idx, corr_idx]
            res_im[blockIdx().x, antenna_idx, corr_idx] += cache[1 + 1 * iq_offset, antenna_idx, corr_idx]
        end
    end
    return nothing
end

function downconvert_and_correlate_kernel_wrapper(
    res_re,
    res_im,
    signal,
    carrier,
    code,
    downconverted_signal,
    correlator_sample_shifts,
    num_samples,
    num_ants,
    num_corrs
)   
    NVTX.@range "kernel params" begin
        # keep num_ants in seperate dimensions, truncate num_samples accordingly to fit
        threads = (512, num_ants)
        blocks = cld(num_samples, 512)
        shmem_size = sizeof(ComplexF32)*512*num_ants*num_corrs
    end
    NVTX.@range "downconvert_and_correlate_kernel" begin
        @cuda threads=threads blocks=blocks shmem=shmem_size downconvert_and_correlate_kernel(
            res_re,
            res_im,
            signal.re,
            signal.im,
            carrier.re,
            carrier.im,
            code,
            downconverted_signal.re,
            downconverted_signal.im,
            correlator_sample_shifts,
            num_samples,
            num_ants,
            num_corrs
    )
    end
    NVTX.@range "sum" begin
        return res_re, res_im
    end
end