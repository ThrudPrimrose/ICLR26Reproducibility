module tsvc_2_s252_mod
  use iso_c_binding
  use omp_lib
  implicit none
contains
  subroutine tsvc_2_s252_fp64(a, b, c, LEN_1D) bind(C, name="tsvc_2_s252_fp64")
    implicit none
    integer(c_int64_t), value :: LEN_1D
    real(c_double), intent(out) :: a(*)
    real(c_double), intent(in) :: b(*), c(*)
    integer(c_int64_t) :: i
    real(c_double), allocatable :: prod(:)

    if (LEN_1D <= 0) return
    allocate(prod(LEN_1D))

    ! Compute elementwise product in parallel
    !$omp parallel do default(none) shared(b, c, prod, LEN_1D) schedule(static)
    do i = 1, LEN_1D
      prod(i) = b(i) * c(i)
    end do
    !$omp end parallel do

    ! First element of a is just the first product
    a(1) = prod(1)

    ! Compute sliding sum a[i] = prod[i] + prod[i-1] for i = 2..LEN_1D in parallel
    if (LEN_1D > 1) then
      !$omp parallel do default(none) shared(prod, a, LEN_1D) schedule(static)
      do i = 2, LEN_1D
        a(i) = prod(i) + prod(i-1)
      end do
      !$omp end parallel do
    end if

    deallocate(prod)
  end subroutine tsvc_2_s252_fp64
end module tsvc_2_s252_mod
