module compact_threshold_pack_mod
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_int64_t), parameter :: CHUNK_SIZE = 32768
    integer(c_int64_t), parameter :: PARALLEL_THRESHOLD = 200000
contains
    subroutine compact_seq(out_count, packed, src, weight, LEN_1D) bind(C, name="compact_seq_helper")
        implicit none
        integer(c_int64_t), value, intent(in) :: LEN_1D
        integer(c_int64_t), intent(inout) :: out_count(1)
        real(c_double), intent(inout) :: packed(LEN_1D)
        real(c_double), intent(in) :: src(LEN_1D)
        real(c_double), intent(in) :: weight(LEN_1D)
        integer(c_int64_t) :: i, n

        n = 0
        do i = 1, LEN_1D
            if (src(i) > 0.0_c_double) then
                packed(n + 1) = src(i) * weight(i)
                n = n + 1
            end if
        end do
        out_count(1) = n
    end subroutine compact_seq

    subroutine compact_par(out_count, packed, src, weight, LEN_1D) bind(C, name="compact_par_helper")
        use omp_lib
        implicit none
        integer(c_int64_t), value, intent(in) :: LEN_1D
        integer(c_int64_t), intent(inout) :: out_count(1)
        real(c_double), intent(inout) :: packed(LEN_1D)
        real(c_double), intent(in) :: src(LEN_1D)
        real(c_double), intent(in) :: weight(LEN_1D)

        integer(c_int64_t) :: num_chunks, c, i, start, end_idx, total, offset, tmp
        integer(c_int64_t), allocatable :: counts(:)
        integer(c_int64_t) :: cnt

        num_chunks = (LEN_1D + CHUNK_SIZE - 1) / CHUNK_SIZE

        allocate(counts(num_chunks))

        !$omp parallel do private(c, i, start, end_idx, cnt) schedule(static)
        do c = 1, num_chunks
            start = (c - 1) * CHUNK_SIZE + 1
            end_idx = min(c * CHUNK_SIZE, LEN_1D)
            cnt = 0
            do i = start, end_idx
                if (src(i) > 0.0_c_double) cnt = cnt + 1
            end do
            counts(c) = cnt
        end do
        !$omp end parallel do

        total = 0
        do c = 1, num_chunks
            tmp = counts(c)
            counts(c) = total
            total = total + tmp
        end do

        !$omp parallel do private(c, i, start, end_idx, offset) schedule(static)
        do c = 1, num_chunks
            start = (c - 1) * CHUNK_SIZE + 1
            end_idx = min(c * CHUNK_SIZE, LEN_1D)
            offset = counts(c)
            do i = start, end_idx
                if (src(i) > 0.0_c_double) then
                    packed(offset + 1) = src(i) * weight(i)
                    offset = offset + 1
                end if
            end do
        end do
        !$omp end parallel do

        out_count(1) = total

        deallocate(counts)
    end subroutine compact_par

    subroutine compact_threshold_pack_fp64(out_count, packed, src, weight, LEN_1D) bind(C, name="compact_threshold_pack_fp64")
        implicit none
        integer(c_int64_t), value, intent(in) :: LEN_1D
        integer(c_int64_t), intent(inout) :: out_count(1)
        real(c_double), intent(inout) :: packed(LEN_1D)
        real(c_double), intent(in) :: src(LEN_1D)
        real(c_double), intent(in) :: weight(LEN_1D)

        if (LEN_1D <= 0) then
            out_count(1) = 0
            return
        end if

        if (LEN_1D < PARALLEL_THRESHOLD) then
            call compact_seq(out_count, packed, src, weight, LEN_1D)
        else
            call compact_par(out_count, packed, src, weight, LEN_1D)
        end if
    end subroutine compact_threshold_pack_fp64
end module compact_threshold_pack_mod
