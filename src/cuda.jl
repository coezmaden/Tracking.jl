struct CUDAConfig
    partial_sum::StructArray
    threads_per_block::Int
    blocks_per_grid::Int
    shmem_size::Int
end

function CUDAConfig(
    num_samples::Int,
    num_ants::NumAnts{NANT},
    num_corrs::Int
) where NANT
    threads_per_block = CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    blocks_per_grid = cld(num_samples, threads_per_block)
    CUDAConfig(
        StructArray{ComplexF32}((CUDA.zeros(Float32, (cld(num_samples, 512), NANT, num_corrs)),CUDA.zeros(Float32, (cld(num_samples, 512), NANT, num_corrs)))),
        threads_per_block,
        blocks_per_grid,
        sizeof(ComplexF32)*NANT*num_corrs
    )
end