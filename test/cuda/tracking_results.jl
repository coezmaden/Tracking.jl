@testset "CUDA: Tracking results" begin
    gpsl1 = GPSL1(use_gpu = Val(true))
    results = Tracking.TrackingResults(
        TrackingState(1, gpsl1, 100Hz, 100, num_samples = 2500),
        EarlyPromptLateCorrelator(NumAnts(2)),
        SVector(-1, 0, 1),
        1,
        1.0Hz,
        1.0,
        true,
        Tracking.BitBuffer(),
        45dBHz
    )

    @test @inferred(get_carrier_doppler(results)) == 100Hz
    @test @inferred(get_carrier_phase(results)) == 0.0
    @test @inferred(get_code_doppler(results)) == 100Hz / 1540
    @test @inferred(get_code_phase(results)) == 100
    @test @inferred(get_correlator_carrier_phase(results)) ≈ 1.0 * 2π
    @test @inferred(get_correlator_carrier_frequency(results)) == 1.0Hz
    correlator = @inferred get_correlator(results)
    @test correlator == EarlyPromptLateCorrelator(NumAnts(2))
    @test @inferred(get_correlator_sample_shifts(results)) == SVector(-1, 0, 1)
    @test @inferred(get_early_late_index_shift(results)) == 1
    @test @inferred(get_early(results)) == [0.0, 0.0]
    @test @inferred(get_prompt(results)) == [0.0, 0.0]
    @test @inferred(get_late(results)) == [0.0, 0.0]
    @test @inferred(get_bits(results)) == 0
    @test @inferred(get_num_bits(results)) == 0
    @test @inferred(get_secondary_code_or_bit_found(results)) == false
    @test @inferred(get_cn0(results)) == 45dBHz

    results = Tracking.TrackingResults(
        TrackingState(1, gpsl1, 100Hz, 100, num_samples = 2500),
        EarlyPromptLateCorrelator(NumAnts(2)),
        SVector(-1, 0, 1),
        1,
        1.0Hz,
        1.0,
        false,
        Tracking.BitBuffer(),
        45dBHz
    )
    correlator = @inferred get_correlator(results)
    @test @inferred(get_early(results)) == [0, 0]
    @test @inferred(get_prompt(results)) == [0, 0]
    @test @inferred(get_late(results)) == [0, 0]
end
