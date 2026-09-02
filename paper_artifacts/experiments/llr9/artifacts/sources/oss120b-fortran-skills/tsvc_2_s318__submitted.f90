subroutine tsvc_2_s318_fp64(a, result, LEN_1D, inc, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  ! Arguments
  integer(c_int64_t), value, intent(in) :: LEN_1D, inc
  real(c_double), intent(in) :: a(*)
  real(c_double), intent(inout) :: result(1)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  ! Locals
  logical, save :: printed = .false.

  integer(c_int64_t) :: i, k, index
  real(c_double) :: maxv, v
  ! Initialize
  k = 0
  index = 0_c_int64_t
  maxv = abs(a(1))
  if (.not. printed) then
    print *, "debug inc=", inc, "LEN=", LEN_1D, "a1=", a(1)
    printed = .true.
  end if
  k = k + inc
  do i = 1, LEN_1D-1
    v = abs(a(k+1))
    if (v > maxv) then
      index = i
      maxv = v
    end if
    k = k + inc
  end do
  result(1) = maxv + dble(index)
end subroutine tsvc_2_s318_fp64
