"""
$(SIGNATURES)

Fixed point CPU StructArray carrier replica generation
"""
function gen_carrier_replica!(
    carrier_replica::StructArray{Complex{T},1,NamedTuple{(:re, :im),Tuple{Array{T,1},Array{T,1}}},Int64},
    carrier_frequency,
    sampling_frequency,
    start_phase,
    carrier_amplitude_power::Val{N},
    start_sample,
    num_samples
) where {
    T <: Integer,
    N
}
    fpcarrier!(
        carrier_replica,
        carrier_frequency,
        sampling_frequency,
        start_phase,
        start_sample = start_sample,
        num_samples = num_samples,
        bits = carrier_amplitude_power
    )
    carrier_replica
end

"""
$(SIGNATURES)

Floating point CPU StructArray carrier generation
"""
function gen_carrier_replica!(
    carrier_replica::StructArray{Complex{T},1,NamedTuple{(:re, :im),Tuple{Array{T,1},Array{T,1}}},Int64},
    carrier_frequency,
    sampling_frequency,
    start_phase,
    carrier_amplitude_power::Val{N},
    start_sample,
    num_samples
) where {
    T <: AbstractFloat,
    N
}
    sample_range = start_sample:num_samples + start_sample - 1
    @views @. carrier_replica.re[sample_range] = 2pi * (sample_range) * carrier_frequency / sampling_frequency + start_phase
    @. carrier_replica.im[sample_range] = sin(carrier_replica.re[sample_range])
    @. carrier_replica.re[sample_range] = cos(carrier_replica.re[sample_range])
    return carrier_replica
end

"""
$(SIGNATURES)

GPU StructArray carrier replica generation
"""
function gen_carrier_replica!(
    carrier_replica::StructArray{Complex{T},1,NamedTuple{(:re, :im),Tuple{CuArray{T,1},CuArray{T,1}}},Int64},
    carrier_frequency,
    sampling_frequency,
    start_phase,
    carrier_amplitude_power,
    start_sample,
    num_samples
) where {
    T <: AbstractFloat
}
    # println("++++ INSIDE gen_carrier_replica! ++++")
    # println("start_sample: ", start_sample)
    # println("num_samples: ", num_samples)
    # println("size of carrier_replica: ", length(carrier_replica))
    sample_range = start_sample:num_samples + start_sample - 1
    @views @. carrier_replica.re[sample_range] = 2pi * (sample_range) * carrier_frequency / sampling_frequency + start_phase
    @. carrier_replica.im[sample_range] = sin(carrier_replica.re[sample_range])
    @. carrier_replica.re[sample_range] = cos(carrier_replica.re[sample_range])
    # println("++++ EXITING gen_carrier_replica! ++++")
    return carrier_replica
end

"""
$(SIGNATURES)

StructArray GPU carrier generation
"""
function gen_carrier_replica_sincos!(
    carrier_replica::StructArray{Complex{T},1,NamedTuple{(:re, :im),Tuple{CuArray{T,1},CuArray{T,1}}},Int64},
    carrier_frequency,
    sampling_frequency,
    start_phase,
    carrier_amplitude_power,
    start_sample,
    num_samples
) where {
    T <: AbstractFloat
}
    @cuda gen_carrier_replica_sincos!(
        carrier_replica.re,
        carrier_replica.im,
        carrier_frequency,
        sampling_frequency,
        start_phase,
        carrier_amplitude_power,
        start_sample,
        num_samples
    )
end

"""
$(SIGNATURES)

sincos testing
"""
function gen_carrier_replica_sincos!(
    carrier_replica_re::CuVector{T},
    carrier_replica_im::CuVector{T},
    carrier_frequency,
    sampling_frequency,
    start_phase,
    carrier_amplitude_power,
    start_sample,
    num_samples
) where {
    T <: AbstractFloat,
}   
    index = threadIdx().x
    stride = blockDim().x
    
    carrier_replica_re, carrier_replica_im = sincos.(2pi .* (start_sample:num_samples) .* carrier_frequency ./ sampling_frequency .+ start_phase)
end

"""
$(SIGNATURES)

GPU complex CuArray carrier generation
"""
function gen_carrier_replica!(
    carrier_replica::CuArray{Complex{T}},
    carrier_frequency,
    sampling_frequency,
    start_phase,
    carrier_amplitude_power,
    start_sample,
    num_samples
) where T <: AbstractFloat
    sample_range = start_sample:num_samples + start_sample - 1
    @. @views carrier_replica[sample_range] = cis(2pi * (0:num_samples-1) * carrier_frequency / sampling_frequency + start_phase)
    return carrier_replica
end

"""
$(SIGNATURES)

Updates the carrier phase.
"""
function update_carrier_phase(
    num_samples,
    carrier_frequency,
    sampling_frequency,
    start_phase,
    carrier_amplitude_power::Val{N}
) where N
    n = N + 2
    fixed_point = 32 - n - 2
    delta = floor(Int32, carrier_frequency * 1 << (fixed_point + n) / sampling_frequency)
    fixed_point_start_phase = floor(Int32, start_phase * 1 << (fixed_point + n))
    phase_fixed_point = delta * num_samples + fixed_point_start_phase
    phase_fixed_point = delta * num_samples + fixed_point_start_phase
    mod(
        phase_fixed_point / 1 << (fixed_point + n) + 0.5,
        1
    ) - 0.5
end

"""
$(SIGNATURES)

Calculates the current carrier frequency.
"""
function get_current_carrier_frequency(intermediate_frequency, carrier_doppler)
    carrier_doppler + intermediate_frequency
end
