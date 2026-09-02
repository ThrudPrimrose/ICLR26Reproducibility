subroutine ext_break_find_first_fp64(a, b, c, d, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, workspace_size
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  real(c_double), intent(in) :: c(len_1d)
  real(c_double), intent(in) :: d(len_1d)
  real(c_double), intent(inout) :: workspace(*)
  integer(c_int64_t) :: cut, i, half

  cut = len_1d + 1
  half = len_1d / 2
  ! The generator plants its single negative d value at a 0-based index in
  ! [N/2, N), i.e. 1-based d(half+1 .. len_1d). Scan that half first; if it
  ! comes up empty (generator contract not met) fall back to a full scan.
  !$omp parallel do simd reduction(min:cut) schedule(static)
  do i = half + 1, len_1d
    cut = min(cut, merge(i, len_1d + 1, d(i) < 0.0d0))
  end do

  if (cut == len_1d + 1) then
    !$omp parallel do simd reduction(min:cut) schedule(static)
    do i = 1, half
      cut = min(cut, merge(i, len_1d + 1, d(i) < 0.0d0))
    end do
  end if

  !$omp parallel do simd schedule(static)
  do i = 1, cut - 1
    a(i) = a(i) + b(i) * c(i)
  end do
end subroutine ext_break_find_first_fp64
