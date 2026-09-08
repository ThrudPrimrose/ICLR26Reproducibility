subroutine tsvc_2_s115_fp64(a, aa, LEN_2D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: a(LEN_2D)
  real(c_double), intent(in) :: aa(LEN_2D, LEN_2D)
  integer(c_int64_t) :: i, j

  ! Outer loop over j from 1 to LEN_2D-1 (since Fortran inclusive)
  do j = 1, LEN_2D - 1
    ! Inner loop over i from j+1 to LEN_2D
    !$omp simd
    do i = j + 1, LEN_2D
      a(i) = a(i) - aa(i, j) * a(j)
    end do
  end do

end subroutine tsvc_2_s115_fp64
