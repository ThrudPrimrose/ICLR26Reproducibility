module tsvc_2_s2233_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s2233_fp64(aa, bb, cc, LEN_2D) bind(C, name="tsvc_2_s2233_fp64")
    real(c_double), intent(inout) :: aa(*)
    real(c_double), intent(inout) :: bb(*)
    real(c_double), intent(in) :: cc(*)
    integer(c_int64_t), value :: LEN_2D
    integer(c_int64_t) :: i, j
    !$omp parallel do schedule(static) default(none) &
    !$omp & shared(aa, cc, LEN_2D) private(i, j)
    do i = 9, LEN_2D
      do j = 9, LEN_2D
        aa((j-1)*LEN_2D + i) = aa((j-2)*LEN_2D + i) + cc((j-1)*LEN_2D + i)
      end do
    end do
    !$omp end parallel do
    do i = 9, LEN_2D
      !$omp parallel do schedule(static) default(none) &
      !$omp & shared(bb, cc, LEN_2D, i) private(j)
      do j = 9, LEN_2D
        bb((i-1)*LEN_2D + j) = bb((i-2)*LEN_2D + j) + cc((i-1)*LEN_2D + j)
      end do
      !$omp end parallel do
    end do
  end subroutine tsvc_2_s2233_fp64
end module tsvc_2_s2233_mod
