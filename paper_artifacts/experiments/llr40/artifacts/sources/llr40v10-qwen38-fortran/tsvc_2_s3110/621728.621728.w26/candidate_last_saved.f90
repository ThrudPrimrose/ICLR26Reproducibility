subroutine tsvc_2_s3110_fp64(aa, bb, LEN_2D, workspace, workspace_size) bind(C, name="tsvc_2_s3110_fp64")
  use iso_c_binding
  implicit none
  real(c_double), dimension(*), intent(in) :: aa
  real(c_double), dimension(*), intent(out) :: bb
  integer(c_int64_t), value, intent(in) :: LEN_2D
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer(c_int64_t) :: i, n2, xindex, yindex
  real(c_double) :: maxv, v, minv
  n2 = LEN_2D*LEN_2D
  write(6, '(A,I0,A,I0)') ' LEN_2D=', LEN_2D, ' n2=', n2
  write(6, *) a16(aa)
  maxv = aa(1); minv = aa(1); xindex = 0; yindex = 0
  do i = 2, n2
    v = aa(i)
    if (v > maxv) then
      maxv = v
      xindex = (i-1)/LEN_2D
      yindex = (i-1) - xindex*LEN_2D
    end if
    if (v < minv) minv = v
  end do
  write(6, '(A,F16.10,A,F16.10,A,I0,A,I0)') ' maxv=', maxv, ' minv=', minv, ' xi=', xindex, ' yi=', yindex
  bb(1) = maxv + real(xindex, c_double) + real(yindex, c_double)
contains
  function a16(a) result(s)
    real(c_double), dimension(*), intent(in) :: a
    real(c_double) :: s(8)
    s = a(1:8)
  end function
end subroutine
