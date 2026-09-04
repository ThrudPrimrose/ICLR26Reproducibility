module tsvc_2_s1232_mod
  use iso_c_binding, only: c_int64_t, c_double
  implicit none
contains
  subroutine tsvc_2_s1232_fp64(aa, bb, cc, LEN_2D, VLEN) bind(c, name='tsvc_2_s1232_fp64')
    integer(c_int64_t), value, intent(in) :: LEN_2D, VLEN
    real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: cc(LEN_2D, LEN_2D)
    integer(c_int64_t) :: i, j, j_max

    !$omp parallel do private(j, j_max)
    do i = 1, LEN_2D
      j_max = min(LEN_2D, (i - 1) / VLEN + 1)
      !$omp simd
      do j = 1, j_max
        aa(j, i) = bb(j, i) + cc(j, i)
      end do
    end do
    !$omp end parallel do
  end subroutine
end module
