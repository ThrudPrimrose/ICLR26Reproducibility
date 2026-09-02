subroutine tsvc_2_s233_fp64(aa, bb, cc, LEN_2D) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(inout) :: bb(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: cc(LEN_2D, LEN_2D)

  integer(c_int64_t), parameter :: B = 118
  integer(c_int64_t) :: i, j, i0, i1

  !$omp parallel private(i, j, i0, i1)
  !$omp do
  do i0 = 9, LEN_2D, B
    i1 = min(i0 + B - 1, LEN_2D)
    do j = 9, LEN_2D
      !$omp simd
      do i = i0, i1
        aa(i, j) = aa(i, j - 1) + cc(i, j)
      end do
      !$omp end simd
    end do
  end do
  !$omp end do nowait

  !$omp do
  do j = 9, LEN_2D
    do i = 9, LEN_2D
      bb(i, j) = bb(i - 1, j) + cc(i, j)
    end do
  end do
  !$omp end do nowait
  !$omp end parallel

end subroutine tsvc_2_s233_fp64
