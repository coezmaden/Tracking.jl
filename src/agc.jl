struct GainControlledSignal{
    S <: Union{StructArray, CuArray},
    A <: Union{Real, Vector{<:Real}}
}
    signal::S
    attenuation::A
    amplitude_power::Int
end

get_signal(agc::GainControlledSignal) = agc.signal
get_attenuation(agc::GainControlledSignal) = agc.attenuation
get_amplitude_power(agc::GainControlledSignal) = agc.amplitude_power

@inline function GainControlledSignal!(
    agc_signal::StructArray{Complex{Int16}},
    signal::AbstractVector,
    bits::Integer = 7
)
    size(agc_signal) == size(signal) ||
        throw(DimensionMismatch("size of AGC signal not equal to size of signal"))
    max_ampl = find_max(signal)
    amplification = 1 << bits / max_ampl
    @inbounds @fastmath for i = eachindex(signal, agc_signal)
        agc_signal.re[i] = floor(Int16, real(signal[i]) * amplification)
        agc_signal.im[i] = floor(Int16, imag(signal[i]) * amplification)
    end
    GainControlledSignal(agc_signal, max_ampl, bits)
end

"""
$(SIGNATURES)

Constructor for a signal to be computed on a GPU
"""
@inline function GainControlledSignal!(
    agc_signal::CuArray{ComplexF32},
    signal::AbstractVector,
    bits::Integer = 7
)
    size(agc_signal) == size(signal) ||
        throw(DimensionMismatch("size of AGC signal not equal to size of signal"))
    agc_signal = signal
    GainControlledSignal(agc_signal, 1, bits)
end

@inline function GainControlledSignal!(
    agc_signal::StructArray{Complex{Int16}},
    signal::AbstractMatrix,
    bits::Integer = 7
)
    size(agc_signal) == size(signal) ||
        throw(DimensionMismatch("size of AGC signal not equal to size of signal"))
    max_ampl = map(find_max, eachcol(signal))
    amplification = (1 << bits) ./ max_ampl
    @inbounds @fastmath for a = axes(signal, 2), i = axes(signal, 1)
        agc_signal.re[i, a] = floor(Int16, real(signal[i, a]) * amplification[a])
        agc_signal.im[i, a] = floor(Int16, imag(signal[i, a]) * amplification[a])
    end
    GainControlledSignal(agc_signal, max_ampl, bits)
end

@inline function GainControlledSignal(signal, bits::Integer = 7)
    GainControlledSignal!(
        StructArray{Complex{Int16}}(undef, size(signal)),
        signal,
        bits
    )
end

@inline function find_max(signal)
    max_real_value = 0.0
    max_imag_value = 0.0
    for i = 1:length(signal)
        if real(signal[i]) > max_real_value
            max_real_value = real(signal[i])
        end
        if imag(signal[i]) > max_imag_value
            max_imag_value = imag(signal[i])
        end
    end
    #sqrt(float(max_real_value)^2 + float(max_imag_value)^2)
    max(max_real_value, max_imag_value)
end
