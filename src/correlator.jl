abstract type AbstractCorrelator{T} end

"""
$(SIGNATURES)

EarlyPromptLateCorrelator for the three correlators: Early, Prompt and Late
"""
struct EarlyPromptLateCorrelator{T} <: AbstractCorrelator{T}
    early::T
    prompt::T
    late::T
end

"""
$(SIGNATURES)

EarlyPromptLateCorrelator constructor without parameters assumes a single antenna.
"""
function EarlyPromptLateCorrelator()
    EarlyPromptLateCorrelator(NumAnts(1))
end

function EarlyPromptLateCorrelator(num_ants::NumAnts{1})
    EarlyPromptLateCorrelator(
        zero(Complex{Float64}),
        zero(Complex{Float64}),
        zero(Complex{Float64})
    )
end

"""
$(SIGNATURES)

EarlyPromptLateCorrelator constructor that considers multiple antennas. The number of
antennas has to be specified by `num_ants::NumAnts{N}` where N is the number of antenna
elements.
"""
function EarlyPromptLateCorrelator(num_ants::NumAnts{N}) where N
    EarlyPromptLateCorrelator(
        zero(SVector{N, Complex{Float64}}),
        zero(SVector{N, Complex{Float64}}),
        zero(SVector{N, Complex{Float64}})
    )
end

"""
$(SIGNATURES)

Get number of antennas from correlator
"""
get_num_ants(correlator::EarlyPromptLateCorrelator{Complex{T}}) where T = 1
get_num_ants(correlator::EarlyPromptLateCorrelator{SVector{N, Complex{T}}}) where {N, T} = N

"""
$(SIGNATURES)

Get the early correlator
"""
@inline get_early(correlator::EarlyPromptLateCorrelator) = correlator.early

"""
$(SIGNATURES)

Get the prompt correlator
"""
@inline get_prompt(correlator::EarlyPromptLateCorrelator) = correlator.prompt

"""
$(SIGNATURES)

Get the late correlator
"""
@inline get_late(correlator::EarlyPromptLateCorrelator) = correlator.late

"""
$(SIGNATURES)

Reset the correlator
"""
function zero(correlator::EarlyPromptLateCorrelator{T}) where T
    EarlyPromptLateCorrelator(zero(T), zero(T), zero(T))
end

"""
$(SIGNATURES)

Filter the correlator by the function `post_corr_filter`.
"""
function filter(post_corr_filter, correlator::EarlyPromptLateCorrelator)
    EarlyPromptLateCorrelator(
        post_corr_filter(get_early(correlator)),
        post_corr_filter(get_prompt(correlator)),
        post_corr_filter(get_late(correlator))
    )
end

"""
$(SIGNATURES)

Calculate the shift between the early and late in samples.
"""
function get_early_late_sample_shift(
    ::Type{S},
    correlator::EarlyPromptLateCorrelator,
    sampling_frequency,
    preferred_code_shift
) where S <: AbstractGNSSSystem
    round(Int, preferred_code_shift * sampling_frequency / get_code_frequency(S))
end

"""
$(SIGNATURES)

Normalize the correlator.
"""
function normalize(correlator::EarlyPromptLateCorrelator, integrated_samples)
    EarlyPromptLateCorrelator(
        get_early(correlator) / integrated_samples,
        get_prompt(correlator) / integrated_samples,
        get_late(correlator) / integrated_samples
    )
end

"""
$(SIGNATURES)

Perform a correlation.
"""
function correlate(
    correlator::EarlyPromptLateCorrelator,
    downconverted_signal,
    code,
    early_late_sample_shift,
    start_sample,
    num_samples_left,
    agc_attenuation,
    agc_bits,
    carrier_bits::Val{NC}
) where NC
    late = zero(Complex{Int32})
    prompt = zero(Complex{Int32})
    early = zero(Complex{Int32})
    @inbounds for i = start_sample:num_samples_left + start_sample - 1
        late = late + downconverted_signal[i] * code[i]
    end
    @inbounds for i = start_sample:num_samples_left + start_sample - 1
        prompt = prompt + downconverted_signal[i] * code[i + early_late_sample_shift]
    end
    @inbounds for i = start_sample:num_samples_left + start_sample - 1
        early = early + downconverted_signal[i] * code[i + 2 * early_late_sample_shift]
    end
    EarlyPromptLateCorrelator(
        get_early(correlator) + early * agc_attenuation / 1 << (agc_bits + NC),
        get_prompt(correlator) + prompt * agc_attenuation / 1 << (agc_bits + NC),
        get_late(correlator) + late * agc_attenuation / 1 << (agc_bits + NC)
    )
end

function correlate(
    correlator::EarlyPromptLateCorrelator{<: SVector{N}},
    downconverted_signal::AbstractMatrix,
    code,
    early_late_sample_shift,
    start_sample,
    num_samples_left,
    agc_attenuation,
    agc_bits,
    carrier_bits::Val{NC}
) where {N, NC}
    late = zero(MVector{N, Complex{Int32}})
    prompt = zero(MVector{N, Complex{Int32}})
    early = zero(MVector{N, Complex{Int32}})
    @inbounds for j = 1:length(late), i = start_sample:num_samples_left + start_sample - 1
        late[j] = late[j] + downconverted_signal[i,j] * code[i]
    end
    @inbounds for j = 1:length(late), i = start_sample:num_samples_left + start_sample - 1
        prompt[j] = prompt[j] + downconverted_signal[i,j] * code[i + early_late_sample_shift]
    end
    @inbounds for j = 1:length(late), i = start_sample:num_samples_left + start_sample - 1
        early[j] = early[j] + downconverted_signal[i,j] * code[i + 2 * early_late_sample_shift]
    end
    EarlyPromptLateCorrelator(
        get_early(correlator) + early .* agc_attenuation / 1 << (agc_bits + NC),
        get_prompt(correlator) + prompt .* agc_attenuation / 1 << (agc_bits + NC),
        get_late(correlator) + late .* agc_attenuation / 1 << (agc_bits + NC)
    )
end

struct VeryEarlyPromptLateCorrelator{T} <: AbstractCorrelator{T}
    very_early::T
    early::T
    prompt::T
    late::T
    very_late::T
end

function VeryEarlyPromptLateCorrelator(num_ants::NumAnts{1})
    VeryEarlyPromptLateCorrelator(
        zero(Complex{Float64}),
        zero(Complex{Float64}),
        zero(Complex{Float64}),
        zero(Complex{Float64}),
        zero(Complex{Float64})
    )
end

function VeryEarlyPromptLateCorrelator(num_ants::NumAnts{N}) where N
    VeryEarlyPromptLateCorrelator(
        zero(SVector{N, Complex{Float64}}),
        zero(SVector{N, Complex{Float64}}),
        zero(SVector{N, Complex{Float64}}),
        zero(SVector{N, Complex{Float64}}),
        zero(SVector{N, Complex{Float64}})
    )
end

"""
$(SIGNATURES)

Get number of antennas from correlator
"""
get_num_ants(correlator::VeryEarlyPromptLateCorrelator{Complex{T}}) where T = 1
function get_num_ants(
    correlator::VeryEarlyPromptLateCorrelator{SVector{N, Complex{T}}}
) where {N, T}
    N
end

@inline get_very_early(correlator::VeryEarlyPromptLateCorrelator) = correlator.early
@inline get_early(correlator::VeryEarlyPromptLateCorrelator) = correlator.early
@inline get_prompt(correlator::VeryEarlyPromptLateCorrelator) = correlator.prompt
@inline get_late(correlator::VeryEarlyPromptLateCorrelator) = correlator.late
@inline get_very_late(correlator::VeryEarlyPromptLateCorrelator) = correlator.late

function zero(correlator::VeryEarlyPromptLateCorrelator{T}) where T
    EarlyPromptLateCorrelator(zero(T), zero(T), zero(T), zero(T), zero(T))
end

function filter(post_corr_filter, correlator::VeryEarlyPromptLateCorrelator)
    EarlyPromptLateCorrelator(
        post_corr_filter(get_very_early(correlator)),
        post_corr_filter(get_early(correlator)),
        post_corr_filter(get_prompt(correlator)),
        post_corr_filter(get_late(correlator)),
        post_corr_filter(get_very_late(correlator))
    )
end

function normalize(correlator::VeryEarlyPromptLateCorrelator, integrated_samples)
    EarlyPromptLateCorrelator(
        get_very_early(correlator) / integrated_samples,
        get_early(correlator) / integrated_samples,
        get_prompt(correlator) / integrated_samples,
        get_late(correlator) / integrated_samples,
        get_very_late(correlator) / integrated_samples
    )
end

# TODO: correlate and dump for very early, very late
#=
Base.@propagate_inbounds @inline function correlate_iteration(
    ::Type{S},
    correlator::VeryEarlyPromptLateCorrelator,
    current_signal,
    early_late_sample_shift,
    carrier,
    prn,
    total_code_length,
    prompt_code_phase
) where S <: AbstractGNSSSystem
    early_code_phase = prompt_code_phase + code_phase_delta * early_late_sample_shift
    early_code_phase += (early_code_phase < 0) * total_code_length
    late_code_phase = prompt_code_phase - code_phase_delta * early_late_sample_shift
    late_code_phase -= (late_code_phase >= total_code_length) * total_code_length
    early_code = get_code_unsafe(S, early_code_phase, prn)
    prompt_code = get_code_unsafe(S, code_phase, prn)
    late_code = get_code_unsafe(S, late_code_phase, prn)
    early = get_early(correlator) + current_signal * carrier * early_code
    prompt = get_prompt(correlator) + current_signal * carrier * prompt_code
    late = get_late(correlator) + current_signal * carrier * late_code
    VeryEarlyPromptLateCorrelator(early, prompt, late)
end
=#

struct VeryVeryEarlyPromptLateCorrelator{T} <: AbstractCorrelator{T}
    veryvery_early::T
    very_early::T
    early::T
    prompt::T
    late::T
    very_late::T
    veryvery_late::T
end

function VeryVeryEarlyPromptLateCorrelator(num_ants::NumAnts{1})
    VeryVeryEarlyPromptLateCorrelator(
        zero(Complex{Float64}),
        zero(Complex{Float64}),
        zero(Complex{Float64}),
        zero(Complex{Float64}),
        zero(Complex{Float64}),
        zero(Complex{Float64}),
        zero(Complex{Float64})
    )
end

function VeryVeryEarlyPromptLateCorrelator(num_ants::NumAnts{N}) where N
    VeryVeryEarlyPromptLateCorrelator(
        zero(SVector{N, Complex{Float64}}),
        zero(SVector{N, Complex{Float64}}),
        zero(SVector{N, Complex{Float64}}),
        zero(SVector{N, Complex{Float64}}),
        zero(SVector{N, Complex{Float64}}),
        zero(SVector{N, Complex{Float64}}),
        zero(SVector{N, Complex{Float64}})
    )
end

"""
$(SIGNATURES)

Get number of antennas from correlator
"""
get_num_ants(correlator::VeryVeryEarlyPromptLateCorrelator{Complex{T}}) where T = 1
function get_num_ants(
    correlator::VeryVeryEarlyPromptLateCorrelator{SVector{N, Complex{T}}}
) where {N, T}
    N
end

@inline get_veryvery_early(correlator::VeryVeryEarlyPromptLateCorrelator) = correlator.early
@inline get_very_early(correlator::VeryVeryEarlyPromptLateCorrelator) = correlator.early
@inline get_early(correlator::VeryVeryEarlyPromptLateCorrelator) = correlator.early
@inline get_prompt(correlator::VeryVeryEarlyPromptLateCorrelator) = correlator.prompt
@inline get_late(correlator::VeryVeryEarlyPromptLateCorrelator) = correlator.late
@inline get_very_late(correlator::VeryVeryEarlyPromptLateCorrelator) = correlator.late
@inline get_veryvery_late(correlator::VeryVeryEarlyPromptLateCorrelator) = correlator.late

function zero(correlator::VeryVeryEarlyPromptLateCorrelator{T}) where T
    VeryVeryEarlyPromptLateCorrelator(zero(T),zero(T),zero(T), zero(T), zero(T), zero(T), zero(T))
end

function filter(post_corr_filter, correlator::VeryVeryEarlyPromptLateCorrelator)
    VeryVeryEarlyPromptLateCorrelator(
        post_corr_filter(get_veryvery_early(correlator)),
        post_corr_filter(get_very_early(correlator)),
        post_corr_filter(get_early(correlator)),
        post_corr_filter(get_prompt(correlator)),
        post_corr_filter(get_late(correlator)),
        post_corr_filter(get_very_late(correlator)),
        post_corr_filter(get_veryvery_late(correlator))
    )
end

function normalize(correlator::VeryVeryEarlyPromptLateCorrelator, integrated_samples)
    VeryVeryEarlyPromptLateCorrelator(
        get_veryvery_early(correlator) / integrated_samples,
        get_very_early(correlator) / integrated_samples,
        get_early(correlator) / integrated_samples,
        get_prompt(correlator) / integrated_samples,
        get_late(correlator) / integrated_samples,
        get_very_late(correlator) / integrated_samples,
        get_veryvery_late(correlator) / integrated_samples,
    )
end


"""
$(SIGNATURES)

Correlator bank for `L` correlators. Assumed to be symmetric, only an odd number of correlators can be initialized.
"""
struct CorrelatorBank{L, N, T} <: AbstractCorrelator{T}
    values::MMatrix{L, N, Complex{T}}
    prompt_index::UInt16
    max_offset::UInt16
end

"""
$(SIGNATURES)

CorrelatorBank constructor with only the `L` input assumes a single antenna
"""
function CorrelatorBank(L::Int)
    iseven(L) ? throw("Number of correlators must be odd") :
    CorrelatorBank(
        MMatrix{L, 1}(zeros(ComplexF64, L, 1)), # values vector
        ceil(UInt16, L/2),                      # prompt index
        UInt16(abs(L - ceil(UInt16, L/2)))      # max offset
    )
end

"""
$(SIGNATURES)

CorrelatorBank constructor that considers multiple antennas. The number of
antennas has to be specified by `num_ants::NumAnts{N}` where N is the number of antenna
elements.
"""
function CorrelatorBank(L::Int, num_ants::NumAnts{N}) where N
    iseven(L) ? throw("Number of correlators must be odd") :
    CorrelatorBank(
        MMatrix{L, N}(zeros(ComplexF64, L, N)), # values vector
        ceil(UInt16, L/2),                      # prompt index
        UInt16(abs(L - ceil(UInt16, L/2)))      # max offset
    )
end

"""
$(SIGNATURES)

Get correlation values from the correlator
"""
get_corr_values(correlator::CorrelatorBank) = correlator.values


"""
$(SIGNATURES)

Get number of antennas from correlator
"""
get_num_ants(correlator::CorrelatorBank{L,N,T}) where {L, N, T} = N

"""
$(SIGNATURES)

Get number of correlators from correlator
"""
get_num_correlators(correlator::CorrelatorBank{L,N,T}) where {L, N, T} = L

"""
$(SIGNATURES)

Get maximum index offset of the `L`-correlator
"""
get_max_offset(correlator::CorrelatorBank) = correlator.max_offset


"""
$(SIGNATURES)

Get the earlyⁿ correlator value. No bound checks.
"""
@inline get_early_unsafe(correlator::CorrelatorBank, n::Int = 1) = @inbounds correlator.values[correlator.prompt_index - n,:]

"""
$(SIGNATURES)

Get the earlyⁿ correlator value. Bound checked.
"""
@inline function get_early(correlator::CorrelatorBank, n::Int = 1)
    if(abs(n) > get_max_offset(correlator))
        throw("Early^$n is out of bounds")
    elseif(n < 0)
        @warn("Used get_early with negative indices. Output is equivalent to get_late(correlator, n)")
    end
    correlator.values[correlator.prompt_index - n,:]
end

"""
$(SIGNATURES)

Get the prompt correlator
"""
@inline get_prompt(correlator::CorrelatorBank) = correlator.values[correlator.prompt_index,:]

"""
$(SIGNATURES)

Get the lateⁿ correlator. No bound checks.
"""
@inline get_late_unsafe(correlator::CorrelatorBank, n::Int = 1) = correlator.values[correlator.prompt_index + n,:]

"""
$(SIGNATURES)

Get the lateⁿ correlator. Bound checked.
"""
@inline function get_late(correlator::CorrelatorBank, n::Int = 1)
    if(abs(n) > get_max_offset(correlator))
        throw("Late^$n is out of bounds")
    elseif(n < 0)
        @warn("Used get_late with negative indices. Output is equivalent to get_early(correlator, n)")
    end
    correlator.values[correlator.prompt_index + n,:]
end

"""
$(SIGNATURES)

Set values of the correlator with a single antenna
"""
function set_values(new_values::Vector, correlator::CorrelatorBank{L,1,T}) where {L, T}
    CorrelatorBank(
        MMatrix{L,1}(new_values),           # values vector
        ceil(UInt16, L/2),                  # prompt index
        UInt16(abs(L - ceil(UInt16, L/2)))  # max offset
    )
end

"""
$(SIGNATURES)

Set values of the correlator with multiple antenna
"""
function set_values(new_values::Matrix, correlator::CorrelatorBank{L,N,T}) where {L, N, T}
    CorrelatorBank(
        MMatrix{L,N}(new_values),           # values vector
        ceil(UInt16, L/2),                  # prompt index
        UInt16(abs(L - ceil(UInt16, L/2)))  # max offset
    )
end

"""
$(SIGNATURES)

Reset the correlator
"""
function zero(correlator::CorrelatorBank{L,N,T}) where {L, N, T}
    CorrelatorBank(L,NumAnts(N))
end

"""
$(SIGNATURES)

Filter the correlator by the function `post_corr_filter`.
"""
function filter(post_corr_filter, correlator::CorrelatorBank{L,N,T}) where {L,N,T}
    filtered_correlator = CorrelatorBank(L,NumAnts(N))
    filtered_correlator.values .= post_corr_filter.(get_corr_values(correlator))
    return filtered_correlator
end

"""
$(SIGNATURES)

Normalize the correlator.
"""
function normalize(correlator::CorrelatorBank{L,N,T}, integrated_samples) where {L,N,T}
    normalized_correlator = CorrelatorBank(L,NumAnts(N))
    normalized_correlator.values .= get_corr_values(correlator) ./ integrated_samples
    return normalized_correlator
end

"""
$(SIGNATURES)

Perform a correlation.
"""
function correlate(
    correlator::CorrelatorBank{L,N,T},
    downconverted_signal,
    code,
    early_late_sample_shift,
    start_sample,
    num_samples_left,
    agc_attenuation,
    agc_bits,
    carrier_bits::Val{NC}
) where {L, N, T, NC}
    corr_values = Vector(undef, get_num_correlators(correlator))

    @inbounds for i in 1:get_num_correlators(correlator)
        @inbounds for n = start_sample:num_samples_left + start_sample - 1 
            corr_values[i] =  downconverted_signal[n] * code[n+(i-1)*early_late_sample_shift]
        end
    end

    return set_values(corr_values,CorrelatorBank(
        get_num_correlators(correlator),
        NumAnts(1)
    ))
end