module tsvc_2_vpvts_mod
  use iso_c_binding
  use omp_lib
contains
  subroutine tsvc_2_vpvts_fp64(a, b, LEN_1D, S) bind(C, name="tsvc_2_vpvts_fp64")
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in) :: b(*)
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t), value, intent(in) :: S
    ! loop variable removed
    integer(c_int) :: tid, nt
    integer(c_int64_t) :: lo, hi
    real(c_double) :: s_val

    s_val = real(S, kind=c_double)

    !$omp parallel private(tid, nt, lo, hi)
    nt = omp_get_num_threads()
    tid = omp_get_thread_num()
    lo = ((LEN_1D * int(tid, c_int64_t)) / int(nt, c_int64_t)) + 1
    hi = ((LEN_1D * int(tid + 1, c_int64_t)) / int(nt, c_int64_t))
    if (lo <= hi) a(lo:hi) = a(lo:hi) + b(lo:hi) * s_val
    !$omp end parallel
  end subroutine tsvc_2_vpvts_fp64
end module tsvc_2_vpvts_mod
