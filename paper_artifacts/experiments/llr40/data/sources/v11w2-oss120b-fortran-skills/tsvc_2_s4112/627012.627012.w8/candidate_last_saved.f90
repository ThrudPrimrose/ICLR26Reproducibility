! Fortran implementation of the tsvc_2_s4112 kernel (double precision)
! The kernel computes: a[i] = a[i] + 2 * b[ip[i]] for i = 0..LEN_1D-1
! The Python reference uses 0-based indexing; Fortran arrays are 1-based, so we offset ip by +1.

subroutine tsvc_2_s4112_fp64(a, b, ip, LEN_1D) bind(C, name="tsvc_2_s4112_fp64")
    use iso_c_binding
    use omp_lib
    implicit none
    ! Scalar length, passed by value
    integer(c_int64_t), value, intent(in) :: LEN_1D
    ! Arrays
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    integer(c_int32_t), intent(in) :: ip(LEN_1D)
    ! Loop index
    integer(c_int64_t) :: i

    !$omp parallel do schedule(static) default(none) shared(a, b, ip, LEN_1D) private(i)
    do i = 1, LEN_1D
        a(i) = a(i) + b(ip(i)) * 2.0d0
    end do
    !$omp end parallel do

end subroutine tsvc_2_s4112_fp64
