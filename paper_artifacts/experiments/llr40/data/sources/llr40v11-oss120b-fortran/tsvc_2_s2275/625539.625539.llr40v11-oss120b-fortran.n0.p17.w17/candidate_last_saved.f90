module tsvc_2_s2275_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s2275_fp64(a, aa, b, bb, c, cc, d, len_2d) bind(C, name="tsvc_2_s2275_fp64")
    implicit none
    integer(c_int64_t), value :: len_2d
    real(c_double), intent(out) :: a(*)
    real(c_double), intent(inout) :: aa(*)
    real(c_double), intent(in) :: b(*)
    real(c_double), intent(in) :: bb(*)
    real(c_double), intent(in) :: c(*)
    real(c_double), intent(in) :: cc(*)
    real(c_double), intent(in) :: d(*)
    integer(c_int64_t) :: i, j, idx
    ! Compute vector a
  !$omp parallel do schedule(static)
  do i = 0, len_2d - 1
    a(i+1) = b(i+1) + c(i+1) * d(i+1)
  end do
  !$omp end parallel do

  ! Update matrix aa
  !$omp parallel do schedule(static)
  do j = 0, len_2d - 1
    !$omp simd
    do i = 0, len_2d - 1
      idx = j * len_2d + i
      aa(idx+1) = aa(idx+1) + bb(idx+1) * cc(idx+1)
    end do
  end do
  !$omp end parallel do
  end subroutine tsvc_2_s2275_fp64
end module tsvc_2_s2275_mod
