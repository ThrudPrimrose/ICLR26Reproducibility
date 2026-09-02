subroutine ext_break_find_first_fp64(a, b, c, d, LEN_1D, workspace, workspace_size) bind(C, name="ext_break_find_first_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D, workspace_size
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D), c(LEN_1D), d(LEN_1D)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t) :: cut, half, i

  if (LEN_1D <= 0) return

  half = LEN_1D / 2
  cut = LEN_1D + 1

  !$omp parallel private(i)
    if (half > 0) then
      !$omp do simd
      do i = 1, half
        a(i) = a(i) + b(i) * c(i)
      end do
    end if

    !$omp do reduction(min:cut)
    do i = half + 1, LEN_1D
      if (d(i) < 0.0d0) then
        cut = i
      end if
    end do

    if (cut > half + 1) then
      !$omp do simd
      do i = half + 1, cut - 1
        a(i) = a(i) + b(i) * c(i)
      end do
    end if
  !$omp end parallel
end subroutine ext_break_find_first_fp64
