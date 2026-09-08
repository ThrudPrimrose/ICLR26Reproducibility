! TSVC s332 find-first-and-capture: first i with a(i) > K (K=1.0),
! out_index/out_value pre-set to sentinels.
! C ABI: ext_break_capture_fp64(const double* a, int64_t* out_index,
!                               double* out_value, int64_t LEN_1D)
subroutine ext_break_capture_fp64(a, out_index, out_value, len_1d) bind(C, name="ext_break_capture_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  real(c_double), intent(in), dimension(*)    :: a
  integer(c_int64_t), intent(out), dimension(*) :: out_index
  real(c_double), intent(out), dimension(*)   :: out_value
  integer(c_int64_t), value, intent(in)       :: len_1d
  integer(c_int64_t) :: i
  real(c_double), parameter :: k = 1.0d0

  out_index(1) = 0_c_int64_t
  out_value(1) = -1.0d0
  do i = 1_c_int64_t, len_1d
    if (a(i) > k) then
      out_index(1) = i
      out_value(1) = a(i)
      exit
    end if
  end do
end subroutine ext_break_capture_fp64
