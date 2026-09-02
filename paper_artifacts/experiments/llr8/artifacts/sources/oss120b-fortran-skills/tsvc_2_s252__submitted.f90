subroutine tsvc_2_s252_fp64(a, b, c, LEN_1D, workspace, workspace_size) bind(C, name="tsvc_2_s252_fp64")
  use iso_c_binding
  implicit none
  ! Arguments
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D)
  real(c_double), intent(in) :: c(LEN_1D)
  type(c_ptr), value :: workspace
  integer(c_int64_t), value :: workspace_size
  integer(c_int64_t) :: i
  real(c_double), allocatable :: prod(:)
  
  if (LEN_1D <= 0) return

    ! Allocate temporary array for products
    allocate(prod(LEN_1D))

  ! Compute products b(i)*c(i)
  !$omp parallel do default(none) shared(b, c, prod, LEN_1D) private(i)
  do i = 1, LEN_1D
    prod(i) = b(i) * c(i)
  end do
  !$omp end parallel do

  ! Compute a(i) from products
  a(1) = prod(1)

  if (LEN_1D > 1_c_int64_t) then
    !$omp parallel do default(none) shared(a, prod, LEN_1D) private(i)
    do i = 2, LEN_1D
      a(i) = prod(i) + prod(i-1)
    end do
    !$omp end parallel do
  end if

  deallocate(prod)

  
end subroutine tsvc_2_s252_fp64
