module tsvc_2_s1244_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s1244_fp64(a, b, c, d, len_1d) bind(C, name="tsvc_2_s1244_fp64")
    implicit none
    integer(c_int64_t), value :: len_1d
    real(c_double), intent(in) :: b(*), c(*)
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(out) :: d(*)
    integer(c_int64_t) :: i
    real(c_double), allocatable :: a_orig(:)
    ! Allocate temporary copy of original a
    allocate(a_orig(len_1d))
    a_orig(1:len_1d) = a(1:len_1d)
    ! Compute a(i)
    !$omp parallel private(i)
    !$omp do schedule(static)
    do i = 0, len_1d - 2
      a(i+1) = b(i+1) + c(i+1) * c(i+1) + b(i+1) * b(i+1) + c(i+1)
    end do
    !$omp end do
    ! Compute d(i) using original a values
        !$omp do schedule(static)
    do i = 0, len_1d - 2
      d(i+1) = a(i+1) + a_orig(i+2)
    end do
    !$omp end do
    !$omp end parallel
    deallocate(a_orig)
  end subroutine tsvc_2_s1244_fp64
end module tsvc_2_s1244_mod
