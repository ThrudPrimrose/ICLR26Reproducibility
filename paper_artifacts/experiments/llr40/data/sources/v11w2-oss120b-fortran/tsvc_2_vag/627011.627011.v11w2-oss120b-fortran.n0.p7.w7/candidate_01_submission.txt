subroutine tsvc_2_vag_fp64(a, b, ip, len_1d) bind(C, name="tsvc_2_vag_fp64")
    use iso_c_binding, only: c_double, c_int32_t, c_int64_t
    implicit none
    real(c_double), intent(out) :: a(*)
    real(c_double), intent(in) :: b(*)
    integer(c_int32_t), intent(in) :: ip(*)
    integer(c_int64_t), value :: len_1d
    integer(c_int64_t) :: i

        do i = 1, len_1d
        a(i) = b(ip(i))
    end do
    end subroutine tsvc_2_vag_fp64
