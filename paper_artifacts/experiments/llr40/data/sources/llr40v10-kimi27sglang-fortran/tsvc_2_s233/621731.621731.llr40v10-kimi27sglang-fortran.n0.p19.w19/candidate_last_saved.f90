subroutine tsvc_2_s233_fp64(aa, bb, cc, LEN_2D) bind(c, name='tsvc_2_s233_fp64')
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(inout) :: bb(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: cc(LEN_2D, LEN_2D)
  integer(c_int64_t) :: i, j, ib, i1
  integer(c_int64_t), parameter :: B = 128
  real(c_double) :: x

  !$omp parallel

  !$omp do schedule(static)
  do ib = 9, LEN_2D, B
    i1 = min(ib + B - 1, LEN_2D)
    do j = 9, LEN_2D
      !$omp simd
      do i = ib, i1
        aa(i, j) = aa(i, j - 1) + cc(i, j)
      end do
    end do
  end do
  !$omp end do nowait

  !$omp do schedule(static) private(x)
  do j = 9, LEN_2D
    x = bb(8, j)
    !$omp simd reduction(inscan, +:x)
    do i = 9, LEN_2D
      x = x + cc(i, j)
      !$omp scan inclusive(x)
      bb(i, j) = x
    end do
  end do
  !$omp end do

  !$omp end parallel
end subroutine tsvc_2_s233_fp64
