subroutine tsvc_2_s235_fp64(a, aa, b, bb, c, len_2d) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: a(len_2d)
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: b(len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d)
  real(c_double), intent(in) :: c(len_2d)
  integer(c_int64_t) :: i, j
  integer :: t, nt
  integer(c_int64_t) :: lo, hi

  !$omp parallel do schedule(static)
  do i = 1, len_2d
    a(i) = a(i) + b(i) * c(i)
  end do

  ! The j-scan chain is per-i only: aa(i,j) reads aa(i,j-1). A fixed i-band
  ! is self-contained for ALL j, so threads own bands with no sync at all.
  !$omp parallel private(t, nt, lo, hi)
  t = omp_get_thread_num()
  nt = omp_get_num_threads()
  lo = (len_2d * t) / nt + 1
  hi = (len_2d * (t + 1)) / nt
  do j = 2, len_2d
    do i = lo, hi
      aa(i, j) = aa(i, j - 1) + bb(i, j) * a(i)
    end do
  end do
  !$omp end parallel
end subroutine tsvc_2_s235_fp64
