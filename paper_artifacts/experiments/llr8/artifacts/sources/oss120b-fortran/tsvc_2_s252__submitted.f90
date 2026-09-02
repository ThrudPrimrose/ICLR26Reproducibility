subroutine tsvc_2_s252_fp64(a, b, c, N, workspace, workspace_bytes) bind(C, name='tsvc_2_s252_fp64')
  use iso_c_binding
  implicit none
  ! Arguments
  real(C_DOUBLE), intent(inout) :: a(*)
  real(C_DOUBLE), intent(in) :: b(*), c(*)
  integer(C_INT64_T), value :: N
  type(C_PTR), value :: workspace      ! unused
  integer(C_INT64_T), value :: workspace_bytes ! unused

  real(C_DOUBLE) :: tmp(N)

  if (N <= 0) return

  ! Compute product into tmp using array syntax (vectorizable)
  tmp = b(1:N) * c(1:N)

  a(1) = tmp(1)
  if (N >= 2) then
    a(2:N) = tmp(2:N) + tmp(1:N-1)
  end if

end subroutine tsvc_2_s252_fp64
