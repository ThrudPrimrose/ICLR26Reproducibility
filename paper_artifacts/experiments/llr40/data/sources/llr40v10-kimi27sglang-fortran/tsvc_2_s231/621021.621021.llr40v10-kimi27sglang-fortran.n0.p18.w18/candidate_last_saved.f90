module tsvc_2_s231_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s231_fp64(aa, bb, LEN_2D) bind(c, name='tsvc_2_s231_fp64')
    integer(c_int64_t), value :: LEN_2D
    real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
    integer(c_int64_t) :: i, j

    !$omp parallel do private(j)
    do i = 1, LEN_2D
      do j = 2, LEN_2D
        aa(i, j) = aa(i, j - 1) + bb(i, j)
      end do
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s231_fp64
end module tsvc_2_s231_mod
