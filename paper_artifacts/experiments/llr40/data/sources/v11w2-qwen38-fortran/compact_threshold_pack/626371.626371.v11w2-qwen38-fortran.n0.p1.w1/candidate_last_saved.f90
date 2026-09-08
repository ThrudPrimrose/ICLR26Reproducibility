! Stream compaction: pack src(i)*weight(i) for every src(i) > 0, publish the count.
! Two-pass: per-thread count -> exclusive prefix -> per-thread compact.
! Pass 2 reads src and weight streaming (product computed unconditionally) to keep
! both input streams sequential; only the store is conditional.
module ctp_mod
    use, intrinsic :: iso_c_binding
    use, intrinsic :: omp_lib
    implicit none
contains

    subroutine parallel_compact(src, weight, packed, N, total)
        integer(c_int64_t), intent(in)  :: N
        integer(c_int64_t), intent(out) :: total
        real(c_double), intent(in)  :: src(1:)
        real(c_double), intent(in)  :: weight(1:)
        real(c_double), intent(out) :: packed(1:)

        integer, allocatable :: counts(:)
        integer(c_int64_t), allocatable :: base(:)
        integer(c_int64_t) :: lo, hi, i, chunk, cc, nl, tt
        real(c_double) :: sv, pr
        integer :: t, nthreads

        nthreads = omp_get_max_threads()
        if (nthreads < 1) nthreads = 1
        if (int(nthreads, 8) > N) nthreads = int(N, 4)

        allocate(counts(nthreads))
        allocate(base(nthreads+1))

        chunk = (N + nthreads - 1) / nthreads

        !$omp parallel default(none) shared(src, weight, packed, N, chunk, counts, base, nthreads) &
        !$omp& private(t, tt, lo, hi, i, cc, nl, sv, pr)
            t = omp_get_thread_num()
            lo  = int(t, 8) * chunk + 1
            hi  = min(N, int(t+1, 8) * chunk)

            ! Pass 1: count survivors in this thread's chunk (streaming read of src).
            cc = 0
            do i = lo, hi
                if (src(i) > 0.0_c_double) cc = cc + 1
            end do
            counts(t+1) = int(cc, 4)
        !$omp barrier
            if (t == 0) then
                base(1) = 0
                do tt = 1, nthreads
                    base(tt+1) = base(tt) + int(counts(tt), 8)
                end do
            end if
        !$omp barrier
            nl = base(t+1)

            ! Pass 2: compact this thread's chunk in order (streaming reads).
            do i = lo, hi
                sv = src(i)
                pr = sv * weight(i)
                if (sv > 0.0_c_double) then
                    packed(nl+1) = pr
                    nl = nl + 1
                end if
            end do
        !$omp end parallel

        total = base(nthreads+1)
        deallocate(counts, base)
    end subroutine parallel_compact
end module ctp_mod


subroutine compact_threshold_pack_fp64(out_count, packed, src, weight, LEN_1D) bind(C, &
&name="compact_threshold_pack_fp64")
    use, intrinsic :: iso_c_binding
    use ctp_mod
    implicit none
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t), intent(inout) :: out_count(1)
    real(c_double), intent(inout) :: packed(LEN_1D)
    real(c_double), intent(in) :: src(LEN_1D)
    real(c_double), intent(in) :: weight(LEN_1D)

    integer(c_int64_t) :: N, tot, i

    N = LEN_1D
    if (N <= 0) then
        out_count(1) = 0
        return
    end if

    if (N < 16384) then
        tot = 0
        do i = 1, N
            if (src(i) > 0.0_c_double) then
                packed(tot+1) = src(i) * weight(i)
                tot = tot + 1
            end if
        end do
        out_count(1) = tot
        return
    end if

    call parallel_compact(src, weight, packed, N, tot)
    out_count(1) = tot
end subroutine compact_threshold_pack_fp64
