module tsvc_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s2233_fp64(aa, bb, cc, LEN_2D) bind(C)
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
    real(c_double), intent(inout) :: bb(LEN_2D, LEN_2D)
    real(c_double), intent(in)    :: cc(LEN_2D, LEN_2D)
    integer(c_int64_t) :: i, j

    ! Parallelize AA across i (columns)
    !$omp parallel do schedule(static) private(j)
    do i = 9, LEN_2D
      do j = 9, LEN_2D
        aa(i, j) = aa(i, j-1) + cc(i, j)
      end do
    end do
    !$omp end parallel do

    ! Parallelize BB across j (rows) with SIMD over i
    !$omp parallel do schedule(static) private(i)
    do j = 9, LEN_2D
      !$omp simd
      do i = 9, LEN_2D
        bb(j, i) = bb(j, i-1) + cc(j, i)
      end do
      !$omp end simd
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s2233_fp64
end module tsvc_mod
