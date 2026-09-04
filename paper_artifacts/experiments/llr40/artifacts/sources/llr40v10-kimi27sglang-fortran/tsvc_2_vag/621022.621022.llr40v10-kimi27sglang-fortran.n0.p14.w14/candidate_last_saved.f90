subroutine tsvc_2_vag_fp64(a, b, ip, LEN_1D) bind(c, name="tsvc_2_vag_fp64")
    use, intrinsic :: iso_c_binding
    implicit none
    integer(c_int64_t), value :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    integer(c_int32_t), intent(in) :: ip(LEN_1D)
    integer(c_int64_t) :: i

    !$omp parallel
    !$omp do simd simdlen(8) schedule(dynamic, 1536)
    do i = 1, LEN_1D
        a(i) = b(ip(i))
    end do
    !$omp end do simd
    !$omp end parallel
end subroutine tsvc_2_vag_fp64
