module tsvc_2_s2275_mod
  use iso_c_binding
  implicit none
contains

  subroutine tsvc_2_s2275_fp64(a, aa, b, bb, c, cc, d, LEN_2D) bind(C, name="tsvc_2_s2275_fp64")
    ! Arguments: a,aa (output), b,bb,c,cc,d (input), LEN_2D (size)
    implicit none
    integer(c_int64_t), value :: LEN_2D
    real(c_double), intent(inout) :: a(LEN_2D)
    real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: b(LEN_2D)
    real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: c(LEN_2D)
    real(c_double), intent(in) :: cc(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: d(LEN_2D)
    integer(c_int64_t) :: i, j

    !$omp parallel do schedule(static) default(none) shared(LEN_2D, a, aa, b, bb, c, cc, d) private(i,j)
    do i = 1, LEN_2D
      do j = 1, LEN_2D
        aa(i, j) = aa(i, j) + bb(i, j) * cc(i, j)
      end do
      a(i) = b(i) + c(i) * d(i)
    end do
    !$omp end parallel do

  end subroutine tsvc_2_s2275_fp64

end module tsvc_2_s2275_mod

