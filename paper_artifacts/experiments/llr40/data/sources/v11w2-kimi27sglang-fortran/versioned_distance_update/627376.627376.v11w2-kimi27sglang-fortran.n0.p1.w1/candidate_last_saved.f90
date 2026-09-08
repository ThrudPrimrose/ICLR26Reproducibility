subroutine versioned_distance_update_fp64(a, b, c, K, LEN_1D, workspace, workspace_bytes) bind(c, name="versioned_distance_update_fp64")
    use iso_c_binding
    use omp_lib
    implicit none
    integer(c_int64_t), value, intent(in) :: LEN_1D, K, workspace_bytes
    type(c_ptr), value, intent(in) :: workspace
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D), c(LEN_1D)
    integer(c_int64_t) :: n, total, blk_len, nblocks, rem
    integer(c_int64_t) :: i, j, m, base, chunk, chunk_size, num_chunks
    integer(c_int64_t) :: j_start, j_end, t64, l, r, nthreads64
    integer :: nthreads, tid
    real(c_double) :: d, dval, start, cl, ol
    real(c_double), allocatable :: coeff(:), offset(:)
    real(c_double), parameter :: coeff75 = 0.75_c_double

    d = coeff75
    n = LEN_1D

    if (n <= K) return

    if (K == 1_c_int64_t) then
        if (n <= 1000_c_int64_t) then
            do i = 2_c_int64_t, n
                a(i) = d * a(i - 1_c_int64_t) + b(i) * c(i)
            end do
            return
        end if

        nthreads = omp_get_max_threads()
        if (nthreads <= 1) then
            do i = 2_c_int64_t, n
                a(i) = d * a(i - 1_c_int64_t) + b(i) * c(i)
            end do
            return
        end if

        nthreads64 = int(nthreads, c_int64_t)
        total = n - 1_c_int64_t
        blk_len = (total + nthreads64 - 1_c_int64_t) / nthreads64

        allocate(coeff(0:nthreads-1), offset(0:nthreads-1))

        !$omp parallel num_threads(nthreads) private(tid, t64, l, r, i, cl, ol, dval)
        tid = omp_get_thread_num()
        t64 = int(tid, c_int64_t)
        l = 1_c_int64_t + t64 * blk_len
        r = min(l + blk_len, n)
        if (l < r) then
            cl = 1.0_c_double
            ol = 0.0_c_double
            do i = l + 1_c_int64_t, r
                dval = b(i) * c(i)
                cl = d * cl
                ol = d * ol + dval
            end do
            coeff(tid) = cl
            offset(tid) = ol
        else
            coeff(tid) = 1.0_c_double
            offset(tid) = 0.0_c_double
        end if
        !$omp end parallel

        do t64 = 1_c_int64_t, nthreads64 - 1_c_int64_t
            cl = coeff(t64)
            coeff(t64) = cl * coeff(t64 - 1_c_int64_t)
            offset(t64) = cl * offset(t64 - 1_c_int64_t) + offset(t64)
        end do

        !$omp parallel num_threads(nthreads) private(tid, t64, l, r, i, start, dval)
        tid = omp_get_thread_num()
        t64 = int(tid, c_int64_t)
        l = 1_c_int64_t + t64 * blk_len
        r = min(l + blk_len, n)
        if (l < r) then
            if (tid == 0) then
                start = a(1)
            else
                start = coeff(tid - 1) * a(1) + offset(tid - 1)
            end if
            dval = b(l + 1_c_int64_t) * c(l + 1_c_int64_t)
            a(l + 1_c_int64_t) = d * start + dval
            do i = l + 2_c_int64_t, r
                dval = b(i) * c(i)
                a(i) = d * a(i - 1_c_int64_t) + dval
            end do
        end if
        !$omp end parallel

        deallocate(coeff, offset)
        return
    end if

    nblocks = (n - K) / K
    rem = n - K - nblocks * K

    if (K <= 8_c_int64_t) then
        !$omp parallel do schedule(static) private(j, i)
        do j = 1_c_int64_t, K
            do i = j + K, n, K
                a(i) = d * a(i - K) + b(i) * c(i)
            end do
        end do
        !$omp end parallel do
    else
        chunk_size = max(8_c_int64_t, K / 64_c_int64_t)
        chunk_size = ((chunk_size + 7_c_int64_t) / 8_c_int64_t) * 8_c_int64_t
        if (chunk_size > K) chunk_size = K
        num_chunks = (K + chunk_size - 1_c_int64_t) / chunk_size

        !$omp parallel do schedule(static) private(chunk, j_start, j_end, m, base, j, i)
        do chunk = 1_c_int64_t, num_chunks
            j_start = (chunk - 1_c_int64_t) * chunk_size + 1_c_int64_t
            j_end = min(j_start + chunk_size - 1_c_int64_t, K)
            do m = 0_c_int64_t, nblocks - 1_c_int64_t
                base = K + m * K
                !$omp simd
                do j = j_start, j_end
                    i = base + j
                    a(i) = d * a(i - K) + b(i) * c(i)
                end do
            end do
        end do
        !$omp end parallel do
    end if

    if (rem > 0_c_int64_t) then
        do i = K + nblocks * K + 1_c_int64_t, n
            a(i) = d * a(i - K) + b(i) * c(i)
        end do
    end if
end subroutine versioned_distance_update_fp64
