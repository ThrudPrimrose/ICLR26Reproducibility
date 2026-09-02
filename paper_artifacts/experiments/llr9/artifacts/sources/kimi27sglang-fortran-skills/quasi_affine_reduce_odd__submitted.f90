subroutine quasi_affine_reduce_odd_fp64(a, out, LEN_1D, workspace, workspace_size) bind(C, name='quasi_affine_reduce_odd_fp64')
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(*)
  real(c_double), intent(out) :: out
  integer(c_int8_t), intent(inout) :: workspace(*)
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer(c_int64_t) :: i, n2, t, nt, lo, hi
  real(c_double) :: acc, local
  integer(c_int64_t), parameter :: threshold = 8192
  real(c_double), allocatable :: parts(:)
  n2 = LEN_1D / 2
  acc = 0.0d0
  if (LEN_1D < threshold) then
    do i = 2, LEN_1D, 2
      acc = acc + a(i)
    end do
  else
    nt = omp_get_max_threads()
    allocate(parts(nt))
    !$omp parallel default(none) shared(parts, a, n2, nt) private(t, lo, hi, i, local)
    t = omp_get_thread_num()
    lo = (n2 * t) / nt + 1
    hi = (n2 * (t + 1)) / nt
    local = 0.0d0
    !$omp simd reduction(+:local)
    do i = lo, hi
      local = local + a(2*i)
    end do
    parts(t + 1) = local
    !$omp end parallel
    do i = 1, nt
      acc = acc + parts(i)
    end do
    deallocate(parts)
  end if
  out = acc
end subroutine quasi_affine_reduce_odd_fp64
