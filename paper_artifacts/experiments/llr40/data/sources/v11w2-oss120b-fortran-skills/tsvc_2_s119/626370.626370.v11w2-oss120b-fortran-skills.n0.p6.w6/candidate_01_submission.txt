module tsvc_2_s119_mod
  use iso_c_binding, only: c_int64_t, c_double
  implicit none
contains
  subroutine tsvc_2_s119_fp64(aa, bb, LEN_2D) bind(C)
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
    integer(c_int64_t) :: i, j
    ! Loop over columns outer, rows inner to achieve unit stride access.
    do j = 2, LEN_2D
      !$omp simd
      do i = 2, LEN_2D
        aa(i, j) = aa(i-1, j-1) + bb(i, j)
      end do
    end do
  end subroutine tsvc_2_s119_fp64
end module tsvc_2_s119_mod
