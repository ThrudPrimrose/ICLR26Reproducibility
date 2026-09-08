module tsvc_2_s252_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s252_fp64(a, b, c, LEN_1D) bind(C, name="tsvc_2_s252_fp64")
    ! Arguments: a (output), b,c (inputs), LEN_1D (size)
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D), c(LEN_1D)
    integer(c_int64_t) :: i

    if (LEN_1D <= 0) return
    ! First element
    a(1) = b(1) * c(1)
    ! Remaining elements: sum of current and previous product, parallelized and vectorized
    !$omp parallel do schedule(static)
    do i = 2, LEN_1D
       a(i) = b(i) * c(i) + b(i-1) * c(i-1)
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s252_fp64
end module tsvc_2_s252_mod
