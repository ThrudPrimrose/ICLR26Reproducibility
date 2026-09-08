module tsvc_2_s1232_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s1232_fp64(aa, bb, cc, LEN_2D, VLEN) bind(C, name="tsvc_2_s1232_fp64")
    ! Arguments
    integer(c_int64_t), value, intent(in) :: LEN_2D
    integer(c_int64_t), value, intent(in) :: VLEN
    real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: cc(LEN_2D, LEN_2D)
    integer(c_int64_t) :: i, j, j_max
    !$omp parallel do default(none) schedule(static) shared(aa, bb, cc, LEN_2D, VLEN) private(i, j_max)
    do i = 1, LEN_2D
      j_max = (i - 1) / VLEN
      do j = 1, j_max + 1
        aa(j, i) = bb(j, i) + cc(j, i)
      end do
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s1232_fp64
end module tsvc_2_s1232_mod
