@testset "CUDA: PLL discriminator" begin

    correlator_minus60off = EarlyPromptLateCorrelator(
        SVector(
            -0.5 + sqrt(3) / 2im,
            -1 + sqrt(3) * 1im,
            -0.5 + sqrt(3) / 2im
        )
    )
    correlator_0off = EarlyPromptLateCorrelator(
        SVector(
            0.5 + 0.0im,
            1 + 0.0im,
            0.5 + 0.0im
        )
    )
    correlator_plus60off = EarlyPromptLateCorrelator(
        SVector(
            0.5 + sqrt(3) / 2im,
            1 + sqrt(3) * 1im,
            0.5 + sqrt(3) / 2im
        )
    )
    gpsl1 = GPSL1(use_gpu = Val(true))
    correlator_sample_shifts = SVector(-1, 0, 1)
    @test @inferred(Tracking.pll_disc(gpsl1, correlator_minus60off, correlator_sample_shifts)) == -π / 3  #-60°
    @test @inferred(Tracking.pll_disc(gpsl1, correlator_0off, correlator_sample_shifts)) == 0
    @test @inferred(Tracking.pll_disc(gpsl1, correlator_plus60off, correlator_sample_shifts)) == π / 3  #+60°
end


@testset "CUDA: DLL discriminator" begin

    sample_shifts = SVector(-2, 0, 2)
    index_shift = 1
    delta = 0.25
    @test @inferred(get_early_late_sample_spacing(sample_shifts, index_shift)) == 4

    gpsl1 = GPSL1(use_gpu = Val(true))
    very_early_correlator = EarlyPromptLateCorrelator(SVector(1.0 + 0.0im, 0.5 + 0.0im, 0.0 + 0.0im))
    early_correlator = EarlyPromptLateCorrelator(SVector(0.75 + 0.0im, 0.75 + 0.0im, 0.25 + 0.0im))
    prompt_correlator = EarlyPromptLateCorrelator(SVector(0.5 + 0.0im, 1 + 0.0im, 0.5 + 0.0im))
    late_correlator = EarlyPromptLateCorrelator(SVector(0.25 + 0.0im, 0.75 + 0.0im, 0.75 + 0.0im))
    very_late_correlator = EarlyPromptLateCorrelator(SVector(0.0 + 0.0im, 0.5 + 0.0im, 1.0 + 0.0im))

    @test @inferred(
        Tracking.dll_disc(gpsl1, very_early_correlator, sample_shifts, index_shift, delta)
    ) == -0.5
    @test @inferred(
        Tracking.dll_disc(gpsl1, early_correlator, sample_shifts, index_shift, delta)
    ) == -0.25
    @test @inferred(
        Tracking.dll_disc(gpsl1, prompt_correlator, sample_shifts, index_shift, delta)
    ) == 0
    @test @inferred(
        Tracking.dll_disc(gpsl1, late_correlator, sample_shifts, index_shift, delta)
    ) == 0.25
    @test @inferred(
        Tracking.dll_disc(gpsl1, very_late_correlator, sample_shifts, index_shift, delta)
    ) == 0.5
end
