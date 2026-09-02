subroutine tsvc_2_s1232_fp64(aa, bb, cc, len_2d, vlen, workspace, workspace_size) bind(C, name="tsvc_2_s1232_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  integer(c_int64_t), value, intent(in) :: vlen
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d)
  real(c_double), intent(in) :: cc(len_2d, len_2d)
  integer(c_int8_t), intent(in) :: workspace(workspace_size)
  integer(c_int64_t) :: i, j

  ! NumPy aa[I,J] (I=row, J=col) lives at flat I*N+J. The Fortran dummy
  ! A(p,q) over those same bytes (col-major) sits at flat (q-1)*N+(p-1),
  ! so A(p,q) = aa[I=q-1, J=p-1]. The reference condition I >= J*VLEN
  ! therefore binds the SECOND subscript. Interchanged so the inner loop
  ! walks the FIRST subscript (unit stride): for row i (=I+1), column j
  ! (=J+1) is touched iff J <= (I)/VLEN, i.e. j <= (i-1)/vlen + 1.
  !$omp parallel do schedule(static, 1)
  do i = 1, len_2d
     do j = 1, (i-1)/vlen + 1
        aa(j, i) = bb(j, i) + cc(j, i)
     end do
  end do
  !$omp end parallel do
end subroutine
