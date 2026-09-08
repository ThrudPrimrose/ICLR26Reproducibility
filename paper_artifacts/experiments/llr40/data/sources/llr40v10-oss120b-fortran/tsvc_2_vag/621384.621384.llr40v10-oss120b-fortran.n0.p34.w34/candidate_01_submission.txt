module tsvc_2_vag_mod
    use iso_c_binding
    implicit none
contains
    subroutine tsvc_2_vag_fp64(a, b, ip, LEN_1D) bind(C, name="tsvc_2_vag_fp64")
        integer(c_int64_t), value :: LEN_1D
        real(c_double), dimension(*), intent(inout) :: a
        real(c_double), dimension(*), intent(in) :: b
        integer(c_int32_t), dimension(*), intent(in) :: ip
        integer(c_int64_t) :: i
        do i = 0_c_int64_t, LEN_1D - 1_c_int64_t
            a(i+1) = b( ip(i+1) )
        end do
    end subroutine tsvc_2_vag_fp64
end module tsvc_2_vag_mod
