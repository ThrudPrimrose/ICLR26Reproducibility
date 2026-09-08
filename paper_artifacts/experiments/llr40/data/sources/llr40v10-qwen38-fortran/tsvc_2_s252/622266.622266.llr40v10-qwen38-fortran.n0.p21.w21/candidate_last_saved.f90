subroutine tsvc_2_s252_fp64(a, b, c, len_1d) bind(c, name='tsvc_2_s252_fp64')
    use, intrinsic :: iso_c_binding
    use, intrinsic :: omp_lib
    implicit none
    integer(c_int64_t), intent(in), value :: len_1d
    real(c_double), intent(out) :: a(*)
    real(c_double), intent(in) :: b(*), c(*)
    integer(c_int64_t) :: n, lo, hi
    integer :: nt, tid

    n = len_1d
    if (n <= 0) return
    if (n == 1) then
        a(1) = b(1)*c(1)
        return
    end if
    a(1) = b(1)*c(1)
    !$omp parallel private(nt, tid, lo, hi)
    nt = omp_get_num_threads()
    tid = omp_get_thread_num()
    lo = 2 + (n - 1) * tid / nt
    hi = 2 + (n - 1) * (tid + 1) / nt
    call work(lo, hi)
    !$omp end parallel

contains

    subroutine work(lo, hi)
        integer(c_int64_t), intent(in) :: lo, hi
        integer(c_int64_t) :: i
        do i = lo, hi - 1
            a(i) = b(i)*c(i) + b(i-1)*c(i-1)
        end do
    end subroutine work
end subroutine
