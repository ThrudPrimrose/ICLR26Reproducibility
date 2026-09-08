subroutine tsvc_2_s1244_fp64(a, b, c, d, LEN_1D) bind(C, name="tsvc_2_s1244_fp64")
    use iso_c_binding
    implicit none
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in) :: b(*), c(*)
    real(c_double), intent(out) :: d(*)
    integer(c_int64_t), value :: LEN_1D
    integer(c_int64_t) :: i
    real(c_double), allocatable :: a_old(:)

    ! Allocate temporary array and copy original a values in parallel.
    allocate(a_old(LEN_1D))
    !$omp parallel do schedule(static)
    do i = 1, LEN_1D
        a_old(i) = a(i)
    end do
    !$omp end parallel do

    !$omp parallel do simd schedule(static)
    do i = 1, LEN_1D-1
        a(i) = b(i) + c(i) * c(i) + b(i) * b(i) + c(i)
        d(i) = a(i) + a_old(i+1)
    end do
    !$omp end parallel do simd

    deallocate(a_old)
end subroutine tsvc_2_s1244_fp64
